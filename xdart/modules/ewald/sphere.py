import logging
import os
from pathlib import Path
from threading import Condition, _PyRLock

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from .arch import EwaldArch
from .arch_series import ArchSeries, _ensure_frames_group
from ssrl_xrd_tools.core.containers import PONI, IntegrationResult1D, IntegrationResult2D
from xdart import utils
from ssrl_xrd_tools.integrate.multi import stitch_1d, stitch_2d


class EwaldSphere:
    """Class for storing multiple arch objects in NeXus-formatted HDF5 files.

    Output file structure::

        scan.hdf5
        ├── entry/                     (NXentry)
        │   ├── calibration/           (PONI geometry)
        │   ├── scan_data/             (motor positions per frame)
        │   ├── frames/                (per-frame NXdata groups)
        │   │   ├── 0001/              (1D result)
        │   │   ├── 0001_2d/           (2D result)
        │   │   ├── 0001_gi1d_*/       (GI 1D results)
        │   │   └── ...
        │   ├── integrated_1d/         (summed 1D, NXdata)
        │   └── integrated_2d/         (summed 2D, NXdata)
        ├── scan_data (legacy DataFrame)
        └── @type = "EwaldSphere"
    """

    def __init__(self, name='scan0', arches=None, data_file=None,
                 scan_data=None, mg_args=None,
                 bai_1d_args=None, bai_2d_args=None,
                 static=False, gi=False, th_mtr=None,
                 incidence_motor=None, geometry=None,
                 series_average=False,
                 overall_raw=0, single_img=False,
                 global_mask=None
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

        self.file_lock = Condition()
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

        self.mg_args = mg_args
        self._mg_integrators = [a.integrator for a in arches] if arches else []

        self.bai_1d_args = bai_1d_args
        self.bai_2d_args = bai_2d_args
        self.sphere_lock = Condition(_PyRLock())

        self.bai_1d: IntegrationResult1D | None = None
        self.bai_2d: IntegrationResult2D | None = None

        self.overall_raw = overall_raw
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
            self.overall_raw = 0

    def add_arch(self, arch=None, calculate=True, update=True, get_sd=True,
                 set_mg=True, h5file=None, batch_save=False, **kwargs):
        """Adds new arch to sphere.

        Parameters
        ----------
        batch_save : bool, default False
            If True, skip the per-frame HDF5 writes of ``scan_data``,
            ``integrated_1d``, and ``integrated_2d`` (all of which delete
            and re-create a grown dataset, so naive per-frame calls cost
            O(N²) over a batch).  The in-memory accumulators are still
            updated, so the caller MUST invoke :meth:`flush_batch_state`
            (or equivalent explicit saves) after the loop to persist them.
        """
        with self.sphere_lock:
            if arch is None:
                arch = EwaldArch(**kwargs)
            if calculate:
                arch.integrate_1d(global_mask=self.global_mask, **self.bai_1d_args)
                arch.integrate_2d(global_mask=self.global_mask, **self.bai_2d_args)
            arch.file_lock = self.file_lock
            self.arches = self.arches.append(pd.Series(arch, index=[arch.idx]),
                                             h5file=h5file,
                                             global_mask=self.global_mask)
            self.arches.sort_index(inplace=True)

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
                if not batch_save:
                    if h5file is not None:
                        self._write_scan_data_nexus(h5file)
                    else:
                        with self.file_lock:
                            with utils.catch_h5py_file(self.data_file, 'a') as file:
                                self._write_scan_data_nexus(file)
            if update:
                self._accumulate_bai_1d(arch)
                if not self.skip_2d:
                    self._accumulate_bai_2d(arch)
                if not batch_save:
                    self.save_bai_1d(h5file=h5file)
                    if not self.skip_2d:
                        self.save_bai_2d(h5file=h5file)
            if set_mg:
                self._mg_integrators = [a.integrator for a in self.arches]

            self.overall_raw += (arch.map_raw - arch.bg_raw)

    def flush_batch_state(self, h5file=None):
        """Persist deferred per-batch state: scan_data + bai_1d/2d.

        Companion to ``add_arch(..., batch_save=True)``.  Writes each of
        ``entry/scan_data``, ``entry/integrated_1d``, and
        ``entry/integrated_2d`` exactly once, so the amortised per-frame
        write cost drops from O(N) HDF5 rewrites to O(1).
        """
        with self.sphere_lock:
            if h5file is not None:
                self._write_scan_data_nexus(h5file)
                self.save_bai_1d(h5file=h5file)
                if not self.skip_2d:
                    self.save_bai_2d(h5file=h5file)
            else:
                with self.file_lock:
                    with utils.catch_h5py_file(self.data_file, 'a') as file:
                        self._write_scan_data_nexus(file)
                        self.save_bai_1d(h5file=file)
                        if not self.skip_2d:
                            self.save_bai_2d(h5file=file)

    def _write_scan_data_nexus(self, h5file):
        """Write scan_data DataFrame into entry/scan_data."""
        entry = h5file.require_group("entry")
        sd_grp = entry.require_group("scan_data")
        sd_grp.attrs.setdefault("NX_class", "NXcollection")
        for col in self.scan_data.columns:
            ds_name = str(col)
            if ds_name in sd_grp:
                del sd_grp[ds_name]
            sd_grp.create_dataset(ds_name, data=self.scan_data[col].values)

    def by_arch_integrate_1d(self, **args):
        """Integrates all arches individually, then sums the results."""
        if not args:
            args = self.bai_1d_args
        else:
            self.bai_1d_args = args.copy()
        with self.sphere_lock:
            self.bai_1d = None
            for arch in self.arches:
                arch.integrate_1d(global_mask=self.global_mask, **args)
                self.arches.__setitem__(arch.idx, arch,
                                        global_mask=self.global_mask)
                self._update_bai_1d(arch)

    def by_arch_integrate_2d(self, **args):
        """Integrates all arches individually, then sums the results."""
        if not args:
            args = self.bai_2d_args
        else:
            self.bai_2d_args = args.copy()
        with self.sphere_lock:
            self.bai_2d = None
            for arch in self.arches:
                arch.integrate_2d(global_mask=self.global_mask, **args)
                self.arches.__setitem__(arch.idx, arch,
                                        global_mask=self.global_mask)
                self._update_bai_2d(arch)

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

    def _update_bai_1d(self, arch, h5file=None):
        """Update running sum of 1D integration results and persist to HDF5."""
        self._accumulate_bai_1d(arch)
        with self.sphere_lock:
            self.save_bai_1d(h5file=h5file)

    def _update_bai_2d(self, arch, h5file=None):
        """Update running sum of 2D integration results and persist to HDF5."""
        self._accumulate_bai_2d(arch)
        if not self.skip_2d:
            with self.sphere_lock:
                self.save_bai_2d(h5file=h5file)

    def set_multi_geo(self, **args):
        """Rebuilds the per-arch integrator list for stitched integration."""
        self.mg_args.update(args)
        with self.sphere_lock:
            self._mg_integrators = [a.integrator for a in self.arches]

    def multigeometry_integrate_1d(self, monitor=None, **kwargs):
        """Stitch all arch images into a single 1D pattern."""
        with self.sphere_lock:
            images = [(a.map_raw - a.bg_raw) for a in self.arches]
            normalization = (
                list(self.scan_data[monitor]) if monitor is not None else None
            )
            return stitch_1d(
                images, self._mg_integrators,
                mask=self.global_mask, normalization=normalization,
                **kwargs,
            )

    def multigeometry_integrate_2d(self, monitor=None, **kwargs):
        """Stitch all arch images into a 2D cake pattern."""
        with self.sphere_lock:
            images = [(a.map_raw - a.bg_raw) / a.map_norm for a in self.arches]
            return stitch_2d(
                images, self._mg_integrators,
                mask=self.global_mask, **kwargs,
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_to_h5(self, replace=False, *args, **kwargs):
        """Saves data to NeXus-formatted hdf5 file."""
        mode = 'w' if replace else 'a'
        with self.file_lock:
            with utils.catch_h5py_file(self.data_file, mode) as file:
                self._save_to_h5(file, *args, **kwargs)

    # ------------------------------------------------------------------
    # v2 NeXus writer (xdart 0.37+).  Sibling of `_save_to_h5`, not a
    # replacement: the v1 path stays until the wrangler is updated to
    # drive the v2 path end-to-end.
    # ------------------------------------------------------------------

    def save_to_nexus(self, *, entry: str = "entry", finalize: bool = False,
                      replace: bool = False) -> None:
        """Save sphere state into a v2 NeXus file.  Idempotent across calls."""
        mode = 'w' if replace else 'a'
        with self.file_lock:
            with utils.catch_h5py_file(self.data_file, mode) as file:
                self._save_to_nexus(file, entry=entry, finalize=finalize)

    def _save_to_nexus(self, h5f, *, entry: str = "entry",
                       finalize: bool = False) -> None:
        """Inner v2 writer; delegates to ``nexus_writer.save_sphere_to_nexus``."""
        from xdart.modules.ewald.nexus_writer import save_sphere_to_nexus
        with self.sphere_lock:
            save_sphere_to_nexus(self, h5f, entry=entry, finalize=finalize)

    def _save_to_h5(self, grp, arches=None, data_only=False,
                    compression=None):
        """Write sphere state to HDF5 in NeXus format."""
        with self.sphere_lock:
            grp.attrs['type'] = 'EwaldSphere'

            entry = grp.require_group("entry")
            entry.attrs.setdefault("NX_class", "NXentry")

            if data_only:
                lst_attr = ["scan_data", "global_mask"]
            else:
                lst_attr = [
                    "scan_data", "global_mask", "mg_args", "bai_1d_args",
                    "bai_2d_args", "overall_raw",
                    "static", "gi", "th_mtr", "single_img",
                    "series_average", "skip_2d"
                ]
            utils.attributes_to_h5(self, grp, lst_attr,
                                   compression=compression)

            if not data_only and hasattr(self, 'arches') and self.arches.index:
                self._write_nexus_calibration(entry)

            if not data_only and not self.scan_data.empty:
                self._write_scan_data_nexus(grp)

            if self.bai_1d is not None:
                if "integrated_1d" in entry:
                    del entry["integrated_1d"]
                self.bai_1d.to_nexus(entry.create_group("integrated_1d"))

            if not self.skip_2d and self.bai_2d is not None:
                if "integrated_2d" in entry:
                    del entry["integrated_2d"]
                self.bai_2d.to_nexus(entry.create_group("integrated_2d"))

    def _write_nexus_calibration(self, entry_grp):
        """Write PONI calibration data into entry/calibration."""
        cal = entry_grp.require_group("calibration")
        cal.attrs.setdefault("NX_class", "NXinstrument")
        try:
            first_arch = self.arches.iloc(0)
            poni = first_arch.poni
        except (IndexError, KeyError):
            return
        if poni is None:
            return
        poni_dict = poni.to_dict()
        for key in ("dist", "poni1", "poni2", "rot1", "rot2", "rot3", "wavelength"):
            if key in cal:
                del cal[key]
            cal.create_dataset(key, data=float(poni_dict.get(key, 0.0)))
        if "detector" in cal:
            del cal["detector"]
        cal.create_dataset("detector", data=str(poni_dict.get("detector", "")))

    def load_from_h5(self, replace=True, mode='r', *args, **kwargs):
        """Loads data from NeXus-formatted hdf5 file."""
        with self.file_lock:
            # Preserve user-supplied global_mask across reset().  v1's
            # _load_from_h5 restored global_mask from the file's root
            # attrs; v2's schema doesn't persist it yet (TODO), so without
            # this preservation the wrangler-provided mask is lost on every
            # post-scan reload and the displayframe shows no mask overlay.
            preserved_global_mask = getattr(self, "global_mask", None)
            if replace:
                self.reset()
                if preserved_global_mask is not None:
                    self.global_mask = preserved_global_mask
            with utils.catch_h5py_file(self.data_file, mode=mode) as file:
                self._load_from_h5(file, *args, **kwargs)

    def _load_from_h5(self, grp, data_only=False, set_mg=True):
        """Load from NeXus-formatted HDF5.

        Auto-detects the schema version:

        * v1 (xdart ≤ 0.36.x): ``grp.attrs["type"] == "EwaldSphere"`` and/or
          digit-named per-frame groups under ``entry/frames/``.
        * v2 (xdart 0.37+): stacked ``entry/integrated_1d (N, nq)``;
          per-frame groups prefixed ``frame_NNNN``.

        Dispatches to :meth:`_load_from_nexus_v2` for v2 files so the
        viewer can open both schemas through a single entry point.
        """
        with self.sphere_lock:
            if _is_v2_layout(grp):
                self._load_from_nexus_v2(grp, data_only=data_only,
                                         set_mg=set_mg)
                return
            if 'type' not in grp.attrs or grp.attrs['type'] != 'EwaldSphere':
                return

            # Build arch index from entry/frames/
            if "entry" in grp and "frames" in grp["entry"]:
                frames = grp["entry/frames"]
                for name in frames:
                    if name.isdigit():
                        idx = int(name)
                        if idx not in self.arches.index:
                            self.arches.index.append(idx)

            self.arches.sort_index(inplace=True)

            if data_only:
                lst_attr = ["scan_data", "overall_raw"]
            else:
                lst_attr = [
                    "scan_data", "mg_args", "bai_1d_args",
                    "bai_2d_args", "overall_raw",
                    "static", "gi", "th_mtr", "single_img",
                    "series_average", "skip_2d"
                ]
            utils.h5_to_attributes(self, grp, lst_attr)

            if not data_only:
                self._set_args(self.bai_1d_args)
                self._set_args(self.bai_2d_args)
                if not self.static:
                    self._set_args(self.mg_args)

                if "entry" in grp and "integrated_1d" in grp["entry"]:
                    self.bai_1d = IntegrationResult1D.from_hdf5(
                        grp["entry/integrated_1d"])
                if "entry" in grp and "integrated_2d" in grp["entry"]:
                    self.bai_2d = IntegrationResult2D.from_hdf5(
                        grp["entry/integrated_2d"])

            if "global_mask" in grp:
                utils.h5_to_attributes(self, grp, ["global_mask"])
            else:
                self.global_mask = None

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
                self.save_to_h5(replace=True, **save_args)
            else:
                if os.path.exists(fname):
                    self.load_from_h5(replace=True, **load_args)
                else:
                    self.reset()
                    self.save_to_h5(replace=True, **save_args)

    def save_bai_1d(self, h5file=None):
        """Save the summed 1D integration result to entry/integrated_1d."""
        if self.bai_1d is None:
            return
        if h5file is not None:
            entry = h5file.require_group("entry")
            if "integrated_1d" in entry:
                del entry["integrated_1d"]
            self.bai_1d.to_nexus(entry.create_group("integrated_1d"))
        else:
            with self.file_lock:
                with utils.catch_h5py_file(self.data_file, 'a') as file:
                    entry = file.require_group("entry")
                    if "integrated_1d" in entry:
                        del entry["integrated_1d"]
                    self.bai_1d.to_nexus(entry.create_group("integrated_1d"))

    def save_bai_2d(self, h5file=None):
        """Save the summed 2D integration result to entry/integrated_2d."""
        if self.bai_2d is None:
            return
        if h5file is not None:
            entry = h5file.require_group("entry")
            if "integrated_2d" in entry:
                del entry["integrated_2d"]
            self.bai_2d.to_nexus(entry.create_group("integrated_2d"))
        else:
            with self.file_lock:
                with utils.catch_h5py_file(self.data_file, 'a') as file:
                    entry = file.require_group("entry")
                    if "integrated_2d" in entry:
                        del entry["integrated_2d"]
                    self.bai_2d.to_nexus(entry.create_group("integrated_2d"))

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
    # v2 NeXus loader  (xdart 0.37+ schema).  Sibling of `_load_from_h5`;
    # called from `_load_from_h5` after schema detection so the viewer's
    # existing entry point handles both v1 and v2 transparently.
    # ------------------------------------------------------------------

    def _load_from_nexus_v2(self, grp, *, data_only: bool = False,
                            set_mg: bool = True) -> None:
        """Populate sphere state from a v2 NXroot.

        Reads via :func:`ssrl_xrd_tools.io.nexus.read_sphere` and
        materialises lightweight ``EwaldArch`` objects from the stacked
        ``intensity_1d`` / ``intensity_2d`` arrays so downstream viewer
        code (which iterates ``self.arches``) needs no schema-specific
        branching.
        """
        from ssrl_xrd_tools.io.nexus import read_sphere
        # IntegrationResult1D/2D, EwaldArch, ArchSeries already imported at
        # module scope; re-import here for clarity / pyright friendliness.

        try:
            ds = read_sphere(self.data_file, schema="v2")
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

        # ── populate arches.index (cheap; required for BOTH data_only paths)
        # In v1, _load_from_h5 builds arches.index from entry/frames/
        # regardless of data_only, because the wrangler's per-frame
        # `update_sphere` flush calls load_from_h5(data_only=True) and
        # the GUI needs the arch indices to update listData.  Mirror
        # that here: populate index first, gate only the heavier
        # work below the data_only check.
        #
        # Lazy-loading: ArchSeries.__getitem__ opens the file in 'r'
        # mode on demand and loads the requested arch from the v1
        # per-frame groups (entry/frames/0001/, 0001_2d/, ...) that
        # the wrangler's add_arch path already wrote to disk.  This
        # avoids the file-lock conflict that would happen if we tried
        # to pass arches into ArchSeries() (which opens in 'a' mode).
        try:
            frame_indices = np.asarray(ds["frame"].values).astype(int).tolist()
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


def _is_v2_layout(grp) -> bool:
    """Detect xdart v2 NeXus layout by inspecting the file-level group.

    True iff a stacked ``entry/integrated_1d/intensity`` of rank 2
    exists, OR ``entry/frames`` contains any ``frame_*`` prefixed
    child.  Mirrors the dispatcher in
    :func:`ssrl_xrd_tools.io.nexus.read_sphere`.
    """
    try:
        if "entry" not in grp:
            return False
        entry = grp["entry"]
        type_attr = entry.attrs.get("type", b"")
        if isinstance(type_attr, bytes):
            type_attr = type_attr.decode("utf-8", errors="replace")
        if type_attr == "EwaldSphere":
            return False
        if "integrated_1d" in entry and "intensity" in entry["integrated_1d"]:
            if entry["integrated_1d"]["intensity"].ndim == 2:
                return True
        if "frames" in entry:
            for name in entry["frames"]:
                if name.startswith("frame_"):
                    return True
    except Exception:
        return False
    return False


def get_1D_data(h5_file, arch_ids=None, static=True):
    """Loads 1D data from NeXus hdf5 file into a Pandas DataFrame."""
    h5_file = Path(h5_file)
    scan_name = h5_file.stem
    sphere = EwaldSphere(scan_name, data_file=str(h5_file), static=static)
    sphere.load_from_h5(replace=False, mode='r')

    rows = []
    with utils.catch_h5py_file(sphere.data_file, 'r') as file:
        if arch_ids is None:
            arch_ids = sphere.arches.index

        frames_grp = file["entry/frames"] if "entry" in file and "frames" in file["entry"] else None

        for idx in arch_ids:
            try:
                arch = EwaldArch(idx=idx, static=sphere.static, gi=sphere.gi)
                if frames_grp is not None:
                    arch.load_from_nexus(frames_grp, load_2d=False)
                else:
                    continue

                if arch.int_1d is None:
                    continue
                _r1d = arch.int_1d
                if _r1d.unit in ('q_A^-1', 'q_nm^-1'):
                    _q = list(_r1d.radial) if _r1d.unit == 'q_A^-1' else list(_r1d.radial / 10.0)
                    _tth = []
                else:
                    _tth = list(_r1d.radial)
                    _q = []
                rows.append({
                    'idx': idx,
                    'intensity': list(_r1d.intensity),
                    'tth': _tth,
                    'q': _q,
                })
            except KeyError:
                pass

    df1 = pd.DataFrame(rows)
    if df1.empty:
        return df1
    df1.set_index('idx', inplace=True)
    df2 = sphere.scan_data

    try:
        df = pd.concat([df1, df2.loc[df1.index]], axis=1, join='outer')
    except KeyError:
        df = df1
    return df
