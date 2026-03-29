# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:21:58 2019

@author: walroth
"""
import copy
from threading import Condition

from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from pyFAI import units
import numpy as np

from xdart import utils
from ssrl_xrd_tools.core.containers import (
    PONI,
    IntegrationResult1D,
    IntegrationResult2D,
)
from xdart.utils.containers.compat import read_legacy_1d, read_legacy_2d
from ssrl_xrd_tools.integrate.calibration import poni_to_integrator
from ssrl_xrd_tools.integrate.single import integrate_1d, integrate_2d
from ssrl_xrd_tools.integrate.gid import (
    create_fiber_integrator,
    integrate_gi_1d,
    integrate_gi_2d,
    integrate_gi_polar,
    integrate_gi_polar_1d,
    integrate_gi_exitangles_1d,
    integrate_gi_exitangles,
)


def _make_integrator_from_poni_dict(poni_dict):
    """Create an AzimuthalIntegrator from a raw poni_dict (legacy AI attributes).

    Extracted so spec_wrangler can build the integrator once per scan
    without constructing a full EwaldArch.
    """
    ai = AzimuthalIntegrator()
    for k, v in poni_dict.items():
        ai.__setattr__(k, v)
    det = getattr(ai, 'detector', None)
    if det is not None and 'MX225' in getattr(det, 'name', ''):
        ai._rot3 -= np.deg2rad(90)
    return ai


class EwaldArch():
    """Class for storing area detector data collected in
    X-ray diffraction experiments.

    Attributes:
        ai_args: dict, arguments passed to AzimuthalIntegrator
        arch_lock: Condition, threading lock used to ensure only one
            process can access data at a time
        file_lock: Condition, lock to ensure only one writer to
            data file
        idx: int, integer name of arch
        int_1d: int_1d_data/_static object from containers (for scanning/static detectors)
        int_2d: int_2d_data/_static object from containers (for scanning/static detectors)
        integrator: AzimuthalIntegrator object from pyFAI
        map_raw: numpy 2d array of the unprocessed image data
        bg_raw: numpy 2d array of the unprocessed image data for BG
        map_norm: float, normalization constant
        mask: numpy array of indeces to be masked in array.
        poni: poni data for integration
        poni_dict: poni_file information saved in dictionary
        scan_info: dict, information from any relevant motors and
            sensors
        static: bool, flag to specify if detector is static
        gi: bool, flag to specify if scattering geometry is grazing incidence
        th_mtr: str, the motor that controls sample rotation in gi mode
        tilt_angle: float, chi offset in gi geometry
        series_average: bool, flag to specify if series of images is averaged

    Methods:
        copy: create copy of arch
        get_mask: return mask array to feed into integrate1d
        integrate_1d: integrate the image data, results stored in
            int_1d_data
        integrate_2d: integrate the image data, results stored in
            int_2d_data
        load_from_h5: load data from hdf5 file
        save_to_h5: save data to hdf5 file
        set_integrator: set new integrator
        set_map_raw: replace raw data
        set_mask: replace mask data
        set_poni: replace poni object
        set_scan_info: replace scan_info
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, idx=None, map_raw=None, poni=None, mask=None,
                 scan_info={}, ai_args={}, file_lock=Condition(),
                 static=False, poni_dict=None, bg_raw=0,
                 gi=False, th_mtr='th', tilt_angle=0,
                 series_average=False, integrator=None
                 ):
        # pylint: disable=too-many-arguments
        """idx: int, name of the arch.
        map_raw: numpy array, raw image data
        bg_raw: numpy array, raw image data for BG
        poni: PONI object, calibration data
        mask: None or numpy array, indices of pixels to mask
        scan_info: dict, metadata about scan
        ai_args: dict, args to be fed to azimuthalIntegrator constructor
        file_lock: Condition, lock for file access.
        """
        super(EwaldArch, self).__init__()
        self.idx = idx
        self.map_raw = map_raw
        self.bg_raw = bg_raw
        if poni is None:
            self.poni = PONI(dist=0.0, poni1=0.0, poni2=0.0)
        else:
            self.poni = poni
        self.poni_dict = poni_dict
        if mask is None and map_raw is not None:
            self.mask = np.arange(map_raw.size)[map_raw.flatten() < 0]
        else:
            self.mask = mask
        self.scan_info = scan_info
        self.ai_args = ai_args
        self.file_lock = file_lock

        self.static = static
        self.gi = gi
        self.th_mtr = th_mtr
        self.tilt_angle = tilt_angle
        self.series_average = series_average

        self.integrator = integrator if integrator is not None else self.setup_integrator()

        self.arch_lock = Condition()
        self.map_norm = 1

        self.int_1d: IntegrationResult1D | None = None
        self.int_2d: IntegrationResult2D | None = None
        self.gi_1d: dict[str, IntegrationResult1D] = {}
        self.gi_2d: dict[str, IntegrationResult2D] = {}

    def __getstate__(self):
        """Exclude threading.Condition objects so EwaldArch can be pickled
        for use with concurrent.futures.ProcessPoolExecutor."""
        state = self.__dict__.copy()
        state.pop('arch_lock', None)
        state.pop('file_lock', None)
        return state

    def __setstate__(self, state):
        """Restore threading.Condition objects after unpickling."""
        self.__dict__.update(state)
        self.arch_lock = Condition()
        self.file_lock = Condition()

    def setup_integrator(self):
        """Sets up integrator object (always a plain AzimuthalIntegrator;
        GI uses create_fiber_integrator transiently during integration)."""
        if self.poni_dict is not None:
            return _make_integrator_from_poni_dict(self.poni_dict)
        else:
            return poni_to_integrator(self.poni)

    def _poni_from_integrator(self):
        """Return an ssrl_xrd_tools PONI derived from self.integrator's current geometry."""
        ai = self.integrator
        det = getattr(ai, 'detector', None)
        det_name = getattr(det, 'name', '') or ''
        wl = getattr(ai, 'wavelength', None)
        return PONI(
            dist=float(ai.dist),
            poni1=float(ai.poni1),
            poni2=float(ai.poni2),
            rot1=float(ai.rot1),
            rot2=float(ai.rot2),
            rot3=float(ai.rot3),
            wavelength=float(wl) if wl else 0.0,
            detector=str(det_name),
        )

    def _get_incident_angle(self):
        """Return incident angle in degrees from th_mtr or scan_info."""
        try:
            return float(self.th_mtr)
        except (ValueError, TypeError):
            return float(self.scan_info.get(self.th_mtr, 0.0))

    def reset(self):
        """Clears all data, resets to a default EwaldArch.
        """
        self.idx = None
        self.map_raw = None
        self.bg_raw = None
        self.poni = PONI()
        # self.poni_dict = None
        self.mask = None
        self.scan_info = {}
        self.integrator = self.setup_integrator()
        self.map_norm = 1
        self.int_1d = None
        self.int_2d = None
        self.gi_1d = {}
        self.gi_2d = {}
            
    def get_mask(self, global_mask=None):
        if global_mask is not None:
            mask_idx = np.unique(np.append(self.mask, global_mask))
            # mask_idx.sort()
        else:
            mask_idx = self.mask
        mask = np.zeros(self.map_raw.size, dtype=int)
        try:
            mask[mask_idx] = 1
            return mask.reshape(self.map_raw.shape)
        except IndexError:
            print('Mask File Shape Mismatch')
            return mask

    def integrate_1d(self, numpoints=10000, radial_range=None,
                     monitor=None, unit=units.TTH_DEG, global_mask=None,
                     fiber_integrator=None, **kwargs):
        """Wrapper for integrate1d method of AzimuthalIntegrator from pyFAI.
        Returns result and also stores the data in the int_1d object.

        args:
            numpoints: int, number of points in final array
            radial_range: tuple or list, lower and upper end of integration
            monitor: str, keyword for normalization counter in scan_info
            unit: pyFAI unit for integration, units.TTH_DEG, units.Q_A,
                '2th_deg', or 'q_A^-1'
            kwargs: other keywords to be passed to integrate1d, see pyFAI docs.

        returns:
            result: integrate1d result from pyFAI.
        """
        with self.arch_lock:
            self.map_norm = 1
            if monitor is not None:
                if monitor.upper() in self.scan_info.keys():
                    self.map_norm = self.scan_info[monitor.upper()]
                elif monitor.lower() in self.scan_info.keys():
                    self.map_norm = self.scan_info[monitor.lower()]

            if self.mask is None:
                self.mask = np.arange(self.map_raw.size)[self.map_raw.flatten() < 0]

            if not self.gi:
                result = integrate_1d(
                    (self.map_raw - self.bg_raw) / self.map_norm,
                    self.integrator,
                    npt=numpoints,
                    unit=str(unit),
                    radial_range=radial_range,
                    mask=self.get_mask(global_mask),
                    **kwargs,
                )
                self.int_1d = result
            else:
                _gi_valid = {
                    'correctSolidAngle', 'variance', 'error_model',
                    'dummy', 'delta_dummy', 'polarization_factor', 'dark', 'flat',
                    'normalization_factor',
                }
                gi_kwargs = {k: v for k, v in kwargs.items() if k in _gi_valid}

                incident_angle = self._get_incident_angle()
                fi = fiber_integrator or create_fiber_integrator(
                    self._poni_from_integrator(),
                    incident_angle=incident_angle,
                    tilt_angle=self.tilt_angle,
                    angle_unit="deg",
                )

                image_data = (self.map_raw - self.bg_raw) / self.map_norm
                mask = self.get_mask(global_mask)
                gi_mode_1d = kwargs.get('gi_mode_1d', 'q_ip')

                # In-plane (Qxy) profile
                r_ip = integrate_gi_1d(
                    image_data, fi, npt=numpoints, unit='qip_A^-1',
                    method='no', mask=mask,
                    radial_range=radial_range,
                    azimuth_range=kwargs.get('azimuth_range'),
                    **gi_kwargs,
                )

                # OOP (Qz) profile: sum the (Qip, Qoop) 2D map over the IP axis
                r2d = integrate_gi_2d(
                    image_data, fi, npt_rad=min(numpoints, 500),
                    npt_azim=min(numpoints, 500),
                    method='no', mask=mask,
                )
                r_qoop = IntegrationResult1D(
                    radial=r2d.azimuthal,
                    intensity=np.nansum(r2d.intensity, axis=0),
                    unit="qoop_A^-1",
                )

                # Q total (polar radial profile)
                r_total = integrate_gi_polar_1d(
                    image_data, fi, npt=numpoints,
                    method='no', mask=mask, **gi_kwargs,
                )

                # Exit angle profile
                r_exit = integrate_gi_exitangles_1d(
                    image_data, fi, npt=numpoints,
                    method='no', mask=mask, **gi_kwargs,
                )

                # Store all GI 1D results
                self.gi_1d = {
                    'qip': r_ip,
                    'qoop': r_qoop,
                    'qtotal': r_total,
                    'exit': r_exit,
                }

                # Set primary result from selected mode
                if gi_mode_1d == 'q_oop':
                    self.int_1d = r_qoop
                elif gi_mode_1d == 'q_total':
                    self.int_1d = r_total
                elif gi_mode_1d == 'exit_angle':
                    self.int_1d = r_exit
                else:  # 'q_ip' (default)
                    self.int_1d = r_ip

    def integrate_2d(self, npt_rad=1000, npt_azim=1000, monitor=None,
                     radial_range=None, azimuth_range=None,
                     x_range=None, y_range=None,
                     unit=units.TTH_DEG, global_mask=None,
                     fiber_integrator=None, **kwargs):
        """Wrapper for integrate2d method of AzimuthalIntegrator from pyFAI.
        Returns result and also stores the data in the int_2d object.

        args:
            npt_rad: int, number of points in radial dimension. If
                None, will take number from the shape of map_norm
            npt_azim: int, number of points in azimuthal dimension. If
                None, will take number from the shape of map_norm
            radial_range: tuple or list, lower and upper end of integration
            azimuth_range: tuple or list, lower and upper end of integration
                in azimuthal direction
            monitor: str, keyword for normalization counter in scan_info
            unit: pyFAI unit for integration, units.TTH_DEG, units.Q_A,
                '2th_deg', or 'q_A^-1'
            x_range: tuple or list, IP range for GI reciprocal map
            y_range: tuple or list, OOP range for GI reciprocal map
            kwargs: other keywords to be passed to integrate2d, see pyFAI docs.

        returns:
            result: integrate2d result from pyFAI.
        """
        with self.arch_lock:
            if monitor is not None:
                self.map_norm = 1

            if self.mask is None:
                self.mask = np.arange(self.map_raw.size)[self.map_raw.flatten() < 0]

            if npt_rad is None:
                npt_rad = self.map_raw.shape[0]

            if npt_azim is None:
                npt_azim = self.map_raw.shape[1]

            if not self.gi:
                result = integrate_2d(
                    (self.map_raw - self.bg_raw) / self.map_norm,
                    self.integrator,
                    npt_rad=npt_rad,
                    npt_azim=npt_azim,
                    unit=str(unit),
                    mask=self.get_mask(global_mask),
                    radial_range=radial_range,
                    azimuth_range=azimuth_range,
                    **kwargs,
                )
                self.int_2d = result
            else:
                _gi_valid = {
                    'correctSolidAngle', 'variance', 'error_model',
                    'dummy', 'delta_dummy', 'polarization_factor', 'dark', 'flat',
                    'normalization_factor',
                }
                gi_kwargs = {k: v for k, v in kwargs.items() if k in _gi_valid}

                incident_angle = self._get_incident_angle()
                fi = fiber_integrator or create_fiber_integrator(
                    self._poni_from_integrator(),
                    incident_angle=incident_angle,
                    tilt_angle=self.tilt_angle,
                    angle_unit="deg",
                )

                image_data = self.map_raw - self.bg_raw
                mask = self.get_mask(global_mask)
                gi_mode_2d = kwargs.get('gi_mode_2d', 'qip_qoop')

                # Polar (Q-Chi) map
                r_polar = integrate_gi_polar(
                    image_data, fi, npt_rad=npt_rad, npt_azim=npt_azim,
                    method='no', mask=mask, **gi_kwargs,
                )

                # Fiber (Qip, Qoop) = (Qxy, Qz) map
                r_gi2d = integrate_gi_2d(
                    image_data, fi, npt_rad=npt_rad, npt_azim=npt_azim,
                    method='no', mask=mask,
                    radial_range=x_range, azimuth_range=y_range,
                    **gi_kwargs,
                )

                # Exit angles map
                r_exit2d = integrate_gi_exitangles(
                    image_data, fi, npt_rad=npt_rad, npt_azim=npt_azim,
                    method='no', mask=mask, **gi_kwargs,
                )

                # Store all GI 2D results in gi_2d dict
                # r_gi2d: radial=Qip, azimuthal=Qoop  (flipud for display convention)
                r_gi2d_flipped = IntegrationResult2D(
                    radial=r_gi2d.radial,
                    azimuthal=r_gi2d.azimuthal,
                    intensity=np.flipud(r_gi2d.intensity),
                    unit=r_gi2d.unit,
                    azimuthal_unit=r_gi2d.azimuthal_unit,
                )
                self.gi_2d = {
                    'polar': r_polar,       # (Q, Chi)
                    'gi2d': r_gi2d_flipped,  # (Qip/Qxy, Qoop/Qz)
                    'exit2d': r_exit2d,     # exit angles
                }

                # Set primary result from selected mode
                if gi_mode_2d == 'q_chi':
                    self.int_2d = r_polar
                elif gi_mode_2d == 'exit_angles':
                    self.int_2d = r_exit2d
                else:  # 'qip_qoop' (default)
                    self.int_2d = r_polar

    def set_integrator(self, **args):
        """Sets AzimuthalIntegrator with new arguments.

        args:
            args: see pyFAI for acceptable arguments for the integrator
                constructor.

        returns:
            None
        """

        with self.arch_lock:
            self.ai_args = args
            self.integrator = self.setup_integrator()

    def set_map_raw(self, new_data):
        with self.arch_lock:
            self.map_raw = new_data
            if self.mask is None:
                self.mask = np.arange(new_data.size)[new_data.flatten() < 0]

    def set_poni(self, new_data):
        with self.arch_lock:
            self.poni = new_data

    def set_mask(self, new_data):
        with self.arch_lock:
            self.mask = new_data

    def set_scan_info(self, new_data):
        with self.arch_lock:
            self.scan_info = new_data

    def save_to_h5(self, file, compression='lzf'):
        """Saves data to hdf5 file using h5py as backend.

        args:
            file: h5py group or file object.

        returns:
            None
        """
        if self.static:
            compression = None
        with self.file_lock:
            if str(self.idx) in file:
                grp = file[str(self.idx)]
            else:
                grp = file.create_group(str(self.idx))
            grp.attrs['type'] = 'EwaldArch'
            if getattr(self, 'skip_map_raw', False):
                lst_attr = [
                    "mask", "map_norm", "scan_info", "ai_args",
                    "gi", "static", "poni_dict"
                ]
            else:
                lst_attr = [
                    "map_raw", "mask", "map_norm", "scan_info", "ai_args",
                    "gi", "static", "poni_dict", "bg_raw"
                ]
            utils.attributes_to_h5(self, grp, lst_attr,
                                   compression=compression)
            if self.int_1d is not None:
                if 'int_1d' not in grp:
                    grp.create_group('int_1d')
                self.int_1d.to_hdf5(grp['int_1d'], compression or "lzf")
            if self.int_2d is not None:
                if 'int_2d' not in grp:
                    grp.create_group('int_2d')
                self.int_2d.to_hdf5(grp['int_2d'], compression or "lzf")
            # Save auxiliary GI results
            if self.gi_1d:
                gi1d_grp = grp.require_group('gi_1d')
                for key, res in self.gi_1d.items():
                    sub = gi1d_grp.require_group(key)
                    res.to_hdf5(sub, compression or "lzf")
            if self.gi_2d:
                gi2d_grp = grp.require_group('gi_2d')
                for key, res in self.gi_2d.items():
                    sub = gi2d_grp.require_group(key)
                    res.to_hdf5(sub, compression or "lzf")
            if 'poni' not in grp:
                grp.create_group('poni')
            utils.dict_to_h5(self.poni.to_dict(), grp, 'poni')

    def load_from_h5(self, file, load_2d=True):
        """Loads data from hdf5 file and sets attributes.

        args:
            file: h5py file or group object
        """
        with self.file_lock:
            with self.arch_lock:
                if str(self.idx) not in file:
                    print("No data can be found")
                else:
                    grp = file[str(self.idx)]
                    if 'type' in grp.attrs:
                        if grp.attrs['type'] == 'EwaldArch':
                            lst_attr = [
                                "map_raw", "mask", "map_norm", "scan_info", "ai_args",
                                "gi", "static", "poni_dict", "bg_raw"
                            ]
                            utils.h5_to_attributes(self, grp, lst_attr)
                            if 'int_1d' in grp:
                                _g1d = grp['int_1d']
                                if 'radial' in _g1d:
                                    self.int_1d = IntegrationResult1D.from_hdf5(_g1d)
                                else:
                                    self.int_1d = read_legacy_1d(_g1d)
                            if load_2d and 'int_2d' in grp:
                                _g2d = grp['int_2d']
                                if 'radial' in _g2d:
                                    self.int_2d = IntegrationResult2D.from_hdf5(_g2d)
                                else:
                                    self.int_2d = read_legacy_2d(_g2d)
                            # Load auxiliary GI results if present
                            if 'gi_1d' in grp:
                                for key in grp['gi_1d']:
                                    self.gi_1d[key] = IntegrationResult1D.from_hdf5(
                                        grp['gi_1d'][key]
                                    )
                            if load_2d and 'gi_2d' in grp:
                                for key in grp['gi_2d']:
                                    self.gi_2d[key] = IntegrationResult2D.from_hdf5(
                                        grp['gi_2d'][key]
                                    )
                            self.poni = PONI.from_dict(
                                utils.h5_to_dict(grp['poni'])
                            )

                            self.integrator = self.setup_integrator()

    def copy(self, include_2d=True):
        """Returns a copy of self.
        """
        arch_copy = EwaldArch(
            copy.deepcopy(self.idx), None,
            copy.deepcopy(self.poni), None,
            copy.deepcopy(self.scan_info), copy.deepcopy(self.ai_args),
            self.file_lock, poni_dict=copy.deepcopy(self.poni_dict),
            static=copy.deepcopy(self.static), gi=copy.deepcopy(self.gi),
            th_mtr=copy.deepcopy(self.th_mtr),
            series_average=copy.deepcopy(self.series_average)
        )
        arch_copy.integrator = copy.deepcopy(self.integrator)
        arch_copy.arch_lock = Condition()
        arch_copy.int_1d = copy.deepcopy(self.int_1d)
        arch_copy.gi_1d = copy.deepcopy(self.gi_1d)
        if include_2d:
            arch_copy.map_raw = copy.deepcopy(self.map_raw)
            arch_copy.bg_raw = copy.deepcopy(self.bg_raw)
            arch_copy.mask = copy.deepcopy(self.mask),
            arch_copy.map_norm = copy.deepcopy(self.map_norm)
            arch_copy.int_2d = copy.deepcopy(self.int_2d)
            arch_copy.gi_2d = copy.deepcopy(self.gi_2d)

        return arch_copy
