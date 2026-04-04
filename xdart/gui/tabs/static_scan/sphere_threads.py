# -*- coding: utf-8 -*-
"""
@author: walroth
"""

# Standard library imports
import logging
from queue import Queue
from threading import Condition
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
import numpy as np

logger = logging.getLogger(__name__)

# Other imports
from xdart.modules.ewald import EwaldArch

# Qt imports
from pyqtgraph import Qt

# This module imports
from xdart.utils import catch_h5py_file as catch
from xdart import utils as ut

import gc



def _reintegrate_arch(arch, bai_1d_args, bai_2d_args, static, gi, do_2d):
    """Module-level worker for parallel reintegration (must be picklable)."""
    from xdart.modules.ewald.arch import EwaldArch  # local import for subprocess safety
    if static:
        arch.static = True
    if gi:
        arch.gi = True
    arch.integrate_1d(**bai_1d_args)
    if do_2d:
        arch.integrate_2d(**bai_2d_args)
    return arch


class integratorThread(Qt.QtCore.QThread):
    """Thread for handling integration. Frees main gui thread from
    intensive calculations.
    
    attributes:
        arch: int, idx of arch to integrate
        lock: Condition, lock to handle access to thread attributes
        method: str, which method to call in run
        mg_1d_args, mg_2d_args: dict, arguments for multigeometry
            integration
        sphere: EwaldSphere, object that does the integration.
    
    methods:
        bai_1d_all: Calls by arch integration 1D for all arches
        bai_1d_SI: Calls by arch integration 1D for specified arch
        bai_2d_all: Calls by arch integration 2D for all arches
        bai_2d_SI: Calls by arch integration 2D for specified arch
        load: Loads data 
        mg_1d: multigeometry 1d integration
        mg_2d: multigeometry 2d integration
        mg_setup: sets up multigeometry object
        run: main thread method.
        
    signals:
        update: empty, tells parent when new data is ready.
    """
    update = Qt.QtCore.Signal(int)

    def __init__(self, sphere, arch, file_lock,
                 arches, arch_ids, data_1d, data_2d,
                 parent=None):
        super().__init__(parent)
        self.sphere = sphere
        self.arch = arch
        self.file_lock = file_lock
        self.arches = arches
        self.arch_ids = arch_ids
        self.data_1d = data_1d
        self.data_2d = data_2d
        self.method = None
        self.lock = Condition()
        self.mg_1d_args = {}
        self.mg_2d_args = {}
    
    def run(self):
        """Calls self.method. Catches exception where method does
        not match any attributes.
        """
        with self.lock:
            method = getattr(self, self.method)
            try:
                method()
            except KeyError as e:
                logger.error("Method %s failed with KeyError: %s", self.method, e, exc_info=True)
                traceback.print_exc()

    def bai_2d_all(self):
        """Integrates all arches 2d. Uses parallel workers when sphere.max_cores > 1."""
        if getattr(self.sphere, 'skip_2d', False):
            return
        self.data_2d.clear()
        with self.sphere.sphere_lock:
            self.sphere.bai_2d = None

        max_cores = getattr(self.sphere, 'max_cores', 1)
        all_arches = list(self.sphere.arches)  # load all from H5 into memory

        if max_cores > 1 and len(all_arches) > 1:
            n_workers = min(max_cores, len(all_arches))
            futures = {}
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                for arch in all_arches:
                    f = executor.submit(
                        _reintegrate_arch, arch,
                        self.sphere.bai_1d_args, self.sphere.bai_2d_args,
                        self.sphere.static, self.sphere.gi, do_2d=True,
                    )
                    futures[f] = arch.idx
                for future in as_completed(futures):
                    try:
                        arch = future.result()
                        self.sphere.arches[arch.idx] = arch
                        self.sphere._update_bai_2d(arch)
                        self.data_2d[int(arch.idx)] = {
                            'map_raw': arch.map_raw, 'bg_raw': arch.bg_raw,
                            'mask': arch.mask, 'int_2d': arch.int_2d, 'gi_2d': arch.gi_2d,
                        }
                        self.update.emit(arch.idx)
                    except Exception as e:
                        arch_idx = futures[future]
                        logger.error("2D integration failed for arch %s: %s", arch_idx, e, exc_info=True)
                        self.update.emit(arch_idx)
        else:
            for arch in all_arches:
                if self.sphere.static:
                    arch.static = True
                if self.sphere.gi:
                    arch.gi = True
                arch.integrate_2d(**self.sphere.bai_2d_args)
                self.sphere.arches[arch.idx] = arch
                self.sphere._update_bai_2d(arch)
                self.data_2d[int(arch.idx)] = {
                    'map_raw': arch.map_raw, 'bg_raw': arch.bg_raw,
                    'mask': arch.mask, 'int_2d': arch.int_2d,
                    'gi_2d': arch.gi_2d,
                }
                self.update.emit(arch.idx)

        with self.file_lock:
            with catch(self.sphere.data_file, 'a') as file:
                ut.dict_to_h5(self.sphere.bai_2d_args, file, 'bai_2d_args')

    def bai_1d_all(self):
        """Integrates all arches 1d. Uses parallel workers when sphere.max_cores > 1."""
        self.data_1d.clear()
        with self.sphere.sphere_lock:
            self.sphere.bai_1d = None

        max_cores = getattr(self.sphere, 'max_cores', 1)
        all_arches = list(self.sphere.arches)  # load all from H5 into memory

        if max_cores > 1 and len(all_arches) > 1:
            n_workers = min(max_cores, len(all_arches))
            futures = {}
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                for arch in all_arches:
                    f = executor.submit(
                        _reintegrate_arch, arch,
                        self.sphere.bai_1d_args, self.sphere.bai_2d_args,
                        self.sphere.static, self.sphere.gi, do_2d=False,
                    )
                    futures[f] = arch.idx
                for future in as_completed(futures):
                    try:
                        arch = future.result()
                        self.sphere.arches[arch.idx] = arch
                        self.sphere._update_bai_1d(arch)
                        self.data_1d[int(arch.idx)] = arch.copy(include_2d=False)
                        self.update.emit(arch.idx)
                    except Exception as e:
                        arch_idx = futures[future]
                        logger.error("1D integration failed for arch %s: %s", arch_idx, e, exc_info=True)
                        self.update.emit(arch_idx)
        else:
            for arch in all_arches:
                if self.sphere.static:
                    arch.static = True
                if self.sphere.gi:
                    arch.gi = True
                arch.integrate_1d(**self.sphere.bai_1d_args)
                self.sphere.arches[arch.idx] = arch
                self.sphere._update_bai_1d(arch)
                self.data_1d[int(arch.idx)] = arch.copy(include_2d=False)
                self.update.emit(arch.idx)

        with self.file_lock:
            with catch(self.sphere.data_file, 'a') as file:
                ut.dict_to_h5(self.sphere.bai_1d_args, file, 'bai_1d_args')

    def bai_2d_SI(self):
        """Integrate the current arch, 2d
        """
        if getattr(self.sphere, 'skip_2d', False):
            return
        idxs = self.arch_ids
        if 'Overall' in self.arch_ids:
            idxs = self.sphere.arches.index
        # for idx in self.arches.keys():
        for idx in idxs:
            # self.sphere.arches[arch].integrate_2d(**self.sphere.bai_2d_args)
            self.sphere.arches[int(idx)].integrate_2d(**self.sphere.bai_2d_args)
            # arch.integrate_2d(**self.sphere.bai_2d_args)
            arch = self.sphere.arches[int(idx)]
            self.data_2d[int(idx)] = {
                'map_raw': arch.map_raw,
                'bg_raw': arch.bg_raw,
                'mask': arch.mask,
                'int_2d': arch.int_2d,
                'gi_2d': arch.gi_2d}
            self.update.emit(idx)

    def bai_1d_SI(self):
        """Integrate the current arch, 1d.
        """
        idxs = self.arch_ids
        if 'Overall' in self.arch_ids:
            idxs = self.sphere.arches.index
        # for (idx, arch) in self.arches.items():
        for idx in idxs:
            # self.sphere.arches[arch].integrate_1d(**self.sphere.bai_1d_args)
            self.sphere.arches[int(idx)].integrate_1d(**self.sphere.bai_1d_args)
            arch = self.sphere.arches[int(idx)]
            self.data_1d[int(arch.idx)] = arch.copy(include_2d=False)
            self.update.emit(arch.idx)

    def load(self):
        """Load data.
        """
        self.sphere.load_from_h5()


class fileHandlerThread(Qt.QtCore.QThread):
    """Thread class for loading data. Handles locks and waiting for
    locks to be released.
    """
    sigNewFile = Qt.QtCore.Signal(str)
    sigUpdate = Qt.QtCore.Signal()
    sigTaskStarted = Qt.QtCore.Signal()
    sigTaskDone = Qt.QtCore.Signal(str)
    
    def __init__(self, sphere, arch, file_lock,
                 parent=None, arch_ids=[], arches=None,
                 data_1d={}, data_2d={}):
        """
        Parameters
        ----------
        file_lock : multiprocessing.Condition
        arch : xdart.modules.ewald.EwaldArch
        sphere : xdart.modules.ewald.EwaldSphere
        """
        super().__init__(parent)
        self.sphere = sphere
        self.arch = arch
        self.arch_ids = arch_ids
        self.arches = arches
        self.data_1d = data_1d
        self.data_2d = data_2d
        self.file_lock = file_lock
        self.queue = Queue()
        self.fname = sphere.data_file
        self.new_fname = None
        self.lock = Condition()
        self.running = False
        self.update_2d = True

    def run(self):
        while True:
            method_name = self.queue.get()
            if method_name is None:
                break  # Sentinel: cleanly exit the thread
            try:
                self.running = True
                self.sigTaskStarted.emit()
                method = getattr(self, method_name)
                method()
            except KeyError as e:
                logger.error("Task %s failed with KeyError: %s", method_name, e, exc_info=True)
                traceback.print_exc()
            self.running = False
            self.sigTaskDone.emit(method_name)
    
    def set_datafile(self):
        with self.file_lock:
            skip_2d = getattr(self.sphere, 'skip_2d', False)
            self.sphere.set_datafile(
                self.fname, save_args={'compression': None}
            )
            self.sphere.skip_2d = skip_2d  # preserve checkbox state across load
        self.sigNewFile.emit(self.fname)
        self.sigUpdate.emit()
    
    def update_sphere(self):
        with self.file_lock:
            try:
                self.sphere.load_from_h5(replace=False, data_only=True,
                                         set_mg=False)
            except KeyError as e:
                logger.debug("Failed to load sphere data from HDF5: %s", e)

    def load_arch(self):
        with self.file_lock:
            with catch(self.sphere.data_file, 'r') as file:
                self.arch.load_from_nexus(file["entry/frames"])
        self.sigUpdate.emit()

    def load_arches(self):
        with self.file_lock:
            with catch(self.sphere.data_file, 'r') as file:
                if "entry" not in file or "frames" not in file["entry"]:
                    return
                frames_grp = file["entry/frames"]

                for idx in self.arch_ids:
                    try:
                        arch = EwaldArch(idx=idx, static=True, gi=self.sphere.gi)
                        arch.load_from_nexus(frames_grp, load_2d=self.update_2d)

                        self.data_1d[int(idx)] = arch.copy(include_2d=False)
                        if self.update_2d:
                            self.data_2d[int(idx)] = {
                                'map_raw': arch.map_raw,
                                'bg_raw': arch.bg_raw,
                                'mask': arch.mask,
                                'int_2d': arch.int_2d,
                                'gi_2d': arch.gi_2d,
                            }

                            if idx in self.arches['add_idxs']:
                                self.arches['sum_int_2d'] += self.data_2d[int(idx)]['int_2d']
                                self.arches['sum_map_raw'] += (self.data_2d[int(idx)]['map_raw'] -
                                                               self.data_2d[int(idx)]['bg_raw'])
                            elif idx in self.arches['sub_idxs']:
                                self.arches['sum_int_2d'] -= self.data_2d[int(idx)]['int_2d']
                                self.arches['sum_map_raw'] -= (self.data_2d[int(idx)]['map_raw'] -
                                                               self.data_2d[int(idx)]['bg_raw'])

                    except KeyError as e:
                        logger.debug("Data missing for arch %s during aggregation: %s", idx, e)

            self.sigUpdate.emit()

        gc.collect()

    def save_data_as(self):
        if self.new_fname is not None and self.new_fname != "":
            with self.file_lock:
                with catch(self.sphere.data_file, 'r') as f1:
                    with catch(self.new_fname, 'w') as f2:
                        for key in f1:
                            f1.copy(key, f2)
                        for attr in f1.attrs:
                            f2.attrs[attr] = f1.attrs[attr]
        self.new_fname = None
