from threading import Condition, _PyRLock
import os
from pathlib import Path
import pandas as pd
import numpy as np

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
                 static=False, gi=False, th_mtr='th', series_average=False,
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
        self.th_mtr = th_mtr
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
                 set_mg=True, h5file=None, **kwargs):
        """Adds new arch to sphere."""
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
                if h5file is not None:
                    self._write_scan_data_nexus(h5file)
                else:
                    with self.file_lock:
                        with utils.catch_h5py_file(self.data_file, 'a') as file:
                            self._write_scan_data_nexus(file)
            if update:
                self._update_bai_1d(arch, h5file=h5file)
                if not self.skip_2d:
                    self._update_bai_2d(arch, h5file=h5file)
            if set_mg:
                self._mg_integrators = [a.integrator for a in self.arches]

            self.overall_raw += (arch.map_raw - arch.bg_raw)

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

    def _update_bai_1d(self, arch, h5file=None):
        """Update running sum of 1D integration results."""
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
            self.save_bai_1d(h5file=h5file)

    def _update_bai_2d(self, arch, h5file=None):
        """Update running sum of 2D integration results."""
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
            if not self.skip_2d:
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
            if replace:
                self.reset()
            with utils.catch_h5py_file(self.data_file, mode=mode) as file:
                self._load_from_h5(file, *args, **kwargs)

    def _load_from_h5(self, grp, data_only=False, set_mg=True):
        """Load from NeXus-formatted HDF5."""
        with self.sphere_lock:
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
