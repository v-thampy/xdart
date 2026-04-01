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

from scipy.ndimage import zoom as ndimage_zoom

from xdart import utils
from ssrl_xrd_tools.core.containers import (
    PONI,
    IntegrationResult1D,
    IntegrationResult2D,
)
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



# Maximum thumbnail dimension — both axes are scaled to fit within this.
_THUMBNAIL_MAX = 256


def _make_thumbnail(image, mask_idx=None, global_mask=None, max_size=_THUMBNAIL_MAX):
    """Downsample a 2D image to at most (max_size, max_size) using scipy zoom.

    Masked pixels are set to NaN *before* downsampling so the mask is
    baked into the thumbnail.  This lets the GUI display a masked preview
    without needing the full-resolution mask array.

    Parameters
    ----------
    image : ndarray
        2D detector image (typically map_raw - bg_raw).
    mask_idx : array-like or None
        Flat indices of pixels to mask (arch-level mask).
    global_mask : array-like or None
        Flat indices of pixels to mask (detector-level mask).
    max_size : int
        Maximum dimension of the output thumbnail.

    Returns float32 array, or None if input is invalid.
    """
    if image is None:
        return None
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 2:
        return None

    # Apply masks as NaN before downsampling
    all_mask = []
    if mask_idx is not None and len(mask_idx) > 0:
        all_mask.append(np.asarray(mask_idx, dtype=int))
    if global_mask is not None and len(global_mask) > 0:
        all_mask.append(np.asarray(global_mask, dtype=int))
    if all_mask:
        flat_mask = np.unique(np.concatenate(all_mask))
        flat_mask = flat_mask[flat_mask < arr.size]  # safety bound
        arr_flat = arr.ravel()
        arr_flat[flat_mask] = np.nan
        arr = arr_flat.reshape(arr.shape)

    h, w = arr.shape
    if h <= max_size and w <= max_size:
        return arr
    factor = min(max_size / h, max_size / w)
    return ndimage_zoom(arr, factor, order=1).astype(np.float32)


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
                 static=False, bg_raw=0,
                 gi=False, th_mtr='th', tilt_angle=0,
                 sample_orientation=4,
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
        self.sample_orientation = sample_orientation
        self.series_average = series_average

        self.integrator = integrator if integrator is not None else self.setup_integrator()
        # Disable pyFAI's legacy mask heuristic that auto-inverts masks
        # with > 50% masked pixels (e.g. threshold masks).
        self.integrator.USE_LEGACY_MASK_NORMALIZATION = False

        self.arch_lock = Condition()
        self.map_norm = 1

        self.int_1d: IntegrationResult1D | None = None
        self.int_2d: IntegrationResult2D | None = None
        self.gi_1d: dict[str, IntegrationResult1D] = {}
        self.gi_2d: dict[str, IntegrationResult2D] = {}
        self.source_file: str = ""
        self.thumbnail: np.ndarray | None = None  # downsampled (map_raw - bg_raw)

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
            # Case-insensitive lookup for motor position
            th_mtr = str(self.th_mtr).lower()
            for key, val in self.scan_info.items():
                if key.lower() == th_mtr:
                    return float(val)
            return 0.0

    def reset(self):
        """Clears all data, resets to a default EwaldArch.
        """
        self.idx = None
        self.map_raw = None
        self.bg_raw = None
        self.poni = PONI(dist=0.0, poni1=0.0, poni2=0.0)
        self.mask = None
        self.scan_info = {}
        self.integrator = self.setup_integrator()
        self.map_norm = 1
        self.int_1d = None
        self.int_2d = None
        self.gi_1d = {}
        self.gi_2d = {}
            
    def get_mask(self, global_mask=None):
        shape = self.map_raw.shape
        size = self.map_raw.size
        mask = np.zeros(size, dtype=bool)
        try:
            if self.mask is not None and len(self.mask):
                mask[self.mask] = True
            if global_mask is not None and len(global_mask):
                mask[global_mask] = True
            return mask.reshape(shape)
        except IndexError:
            print('Mask File Shape Mismatch')
            return mask.reshape(shape) if mask.size == size else mask

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
            if self.map_raw is None:
                return  # no raw data (e.g. loaded from NeXus) — cannot re-integrate

            self.map_norm = 1
            if monitor is not None:
                if monitor.upper() in self.scan_info.keys():
                    self.map_norm = self.scan_info[monitor.upper()]
                elif monitor.lower() in self.scan_info.keys():
                    self.map_norm = self.scan_info[monitor.lower()]

            if self.mask is None:
                self.mask = np.arange(self.map_raw.size)[self.map_raw.flatten() < 0]

            # Keys that only make sense for GI integration — strip before
            # passing to the standard pyFAI integrator.
            _gi_only_keys = {
                'gi_mode_1d', 'gi_mode_2d', 'npt_oop',
                'sample_orientation', 'tilt_angle',
            }

            if not self.gi:
                std_kwargs = {k: v for k, v in kwargs.items()
                              if k not in _gi_only_keys}
                chi_offset = std_kwargs.pop('chi_offset', 0.0)
                if 'azimuth_range' in std_kwargs and std_kwargs['azimuth_range'] is not None:
                    _az = std_kwargs['azimuth_range']
                    std_kwargs['azimuth_range'] = (_az[0] - chi_offset, _az[1] - chi_offset)
                
                result = integrate_1d(
                    (self.map_raw - self.bg_raw) / self.map_norm,
                    self.integrator,
                    npt=numpoints,
                    unit=str(unit),
                    radial_range=radial_range,
                    mask=self.get_mask(global_mask),
                    **std_kwargs,
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
                    sample_orientation=self.sample_orientation,
                    angle_unit="deg",
                )
                self._fiber_integrator = fi  # cache for integrate_2d

                image_data = (self.map_raw - self.bg_raw) / self.map_norm
                mask = self.get_mask(global_mask)
                gi_mode_1d = kwargs.get('gi_mode_1d', 'q_total')
                npt_oop = kwargs.get('npt_oop', numpoints)

                # Only compute the selected GI 1D mode
                if gi_mode_1d == 'q_ip':
                    result = integrate_gi_1d(
                        image_data, fi, npt=numpoints, npt_oop=npt_oop,
                        unit='qip_A^-1',
                        method='no', mask=mask,
                        radial_range=radial_range,
                        azimuth_range=kwargs.get('azimuth_range'),
                        **gi_kwargs,
                    )
                    self.gi_1d['qip'] = result
                elif gi_mode_1d == 'q_oop':
                    # OOP (Qz) profile: sum the (Qip, Qoop) 2D map over IP
                    r2d = integrate_gi_2d(
                        image_data, fi, npt_rad=numpoints,
                        npt_azim=npt_oop,
                        method='no', mask=mask,
                        radial_range=radial_range,
                        azimuth_range=kwargs.get('azimuth_range'),
                    )
                    result = IntegrationResult1D(
                        radial=r2d.azimuthal,
                        intensity=np.nansum(r2d.intensity, axis=0),
                        unit="qoop_A^-1",
                    )
                    self.gi_1d['qoop'] = result
                elif gi_mode_1d == 'exit_angle':
                    result = integrate_gi_exitangles_1d(
                        image_data, fi, npt=numpoints,
                        method='no', mask=mask, 
                        radial_range=radial_range,
                        azimuth_range=kwargs.get('azimuth_range'),
                        **gi_kwargs,
                    )
                    self.gi_1d['exit'] = result
                else:  # 'q_total' (default — polar integration)
                    result = integrate_gi_polar_1d(
                        image_data, fi, npt=numpoints,
                        method='no', mask=mask,
                        radial_range=radial_range,
                        azimuth_range=kwargs.get('azimuth_range'),
                        **gi_kwargs,
                    )
                    self.gi_1d['qtotal'] = result

                self.int_1d = result

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
            if self.map_raw is None:
                return  # no raw data (e.g. loaded from NeXus) — cannot re-integrate

            if monitor is not None:
                self.map_norm = 1

            if self.mask is None:
                self.mask = np.arange(self.map_raw.size)[self.map_raw.flatten() < 0]

            if npt_rad is None:
                npt_rad = self.map_raw.shape[0]

            if npt_azim is None:
                npt_azim = self.map_raw.shape[1]

            _gi_only_keys = {
                'gi_mode_1d', 'gi_mode_2d', 'npt_oop',
                'sample_orientation', 'tilt_angle',
            }

            if not self.gi:
                std_kwargs = {k: v for k, v in kwargs.items()
                              if k not in _gi_only_keys}
                chi_offset = std_kwargs.pop('chi_offset', 0.0)
                if azimuth_range is not None:
                    azimuth_range = (azimuth_range[0] - chi_offset, azimuth_range[1] - chi_offset)
                
                result = integrate_2d(
                    (self.map_raw - self.bg_raw) / self.map_norm,
                    self.integrator,
                    npt_rad=npt_rad,
                    npt_azim=npt_azim,
                    unit=str(unit),
                    mask=self.get_mask(global_mask),
                    radial_range=radial_range,
                    azimuth_range=azimuth_range,
                    **std_kwargs,
                )
                
                # Apply explicit chi_offset rather than guessing midpoint
                if len(result.azimuthal) > 0:
                    result.azimuthal = result.azimuthal + chi_offset
                self.int_2d = result
            else:
                _gi_valid = {
                    'correctSolidAngle', 'variance', 'error_model',
                    'dummy', 'delta_dummy', 'polarization_factor', 'dark', 'flat',
                    'normalization_factor',
                }
                gi_kwargs = {k: v for k, v in kwargs.items() if k in _gi_valid}

                # Re-use FiberIntegrator cached by integrate_1d when available
                fi = fiber_integrator or getattr(self, '_fiber_integrator', None)
                if fi is None:
                    incident_angle = self._get_incident_angle()
                    fi = create_fiber_integrator(
                        self._poni_from_integrator(),
                        incident_angle=incident_angle,
                        tilt_angle=self.tilt_angle,
                        sample_orientation=self.sample_orientation,
                        angle_unit="deg",
                    )

                image_data = (self.map_raw - self.bg_raw) / self.map_norm
                mask = self.get_mask(global_mask)
                gi_mode_2d = kwargs.get('gi_mode_2d', 'qip_qoop')

                # Only compute the selected GI 2D mode
                if gi_mode_2d == 'q_chi':
                    result = integrate_gi_polar(
                        image_data, fi, npt_rad=npt_rad, npt_azim=npt_azim,
                        method='no', mask=mask,
                        radial_range=radial_range,
                        azimuth_range=azimuth_range,
                        **gi_kwargs,
                    )
                    self.gi_2d['polar'] = result
                elif gi_mode_2d == 'exit_angles':
                    result = integrate_gi_exitangles(
                        image_data, fi, npt_rad=npt_rad, npt_azim=npt_azim,
                        method='no', mask=mask,
                        radial_range=radial_range,
                        azimuth_range=azimuth_range,
                        **gi_kwargs,
                    )
                    self.gi_2d['exit2d'] = result
                else:  # 'qip_qoop' (default)
                    r_gi2d = integrate_gi_2d(
                        image_data, fi, npt_rad=npt_rad, npt_azim=npt_azim,
                        method='no', mask=mask,
                        radial_range=x_range, azimuth_range=y_range,
                        **gi_kwargs,
                    )
                    # integrate_gi_2d returns (npt_ip, npt_oop) via
                    # _to_result_2d transpose.  Use directly — the display
                    # code skips the azimuthal flip for qip_qoop mode.
                    result = r_gi2d
                    self.gi_2d['gi2d'] = result

                self.int_2d = result

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

    def save_to_h5(self, file, compression=None):
        """Saves data to hdf5 file using h5py as backend.
        DEPRECATED — use save_to_nexus for new code.

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
                    "gi", "static"
                ]
            else:
                lst_attr = [
                    "map_raw", "mask", "map_norm", "scan_info", "ai_args",
                    "gi", "static", "bg_raw"
                ]
            utils.attributes_to_h5(self, grp, lst_attr,
                                   compression=compression)
            if self.int_1d is not None:
                if 'int_1d' not in grp:
                    grp.create_group('int_1d')
                self.int_1d.to_hdf5(grp['int_1d'], compression)
            if self.int_2d is not None:
                if 'int_2d' not in grp:
                    grp.create_group('int_2d')
                self.int_2d.to_hdf5(grp['int_2d'], compression)
            # Save auxiliary GI results
            if self.gi_1d:
                gi1d_grp = grp.require_group('gi_1d')
                for key, res in self.gi_1d.items():
                    sub = gi1d_grp.require_group(key)
                    res.to_hdf5(sub, compression)
            if self.gi_2d:
                gi2d_grp = grp.require_group('gi_2d')
                for key, res in self.gi_2d.items():
                    sub = gi2d_grp.require_group(key)
                    res.to_hdf5(sub, compression)
            if 'poni' not in grp:
                grp.create_group('poni')
            utils.dict_to_h5(self.poni.to_dict(), grp, 'poni')

    def load_from_h5(self, file, load_2d=True):
        """Loads data from hdf5 file and sets attributes.
        DEPRECATED — use load_from_nexus for new code.

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
                                "gi", "static", "bg_raw"
                            ]
                            utils.h5_to_attributes(self, grp, lst_attr)
                            if 'int_1d' in grp:
                                self.int_1d = IntegrationResult1D.from_hdf5(grp['int_1d'])
                            if load_2d and 'int_2d' in grp:
                                self.int_2d = IntegrationResult2D.from_hdf5(grp['int_2d'])
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

    # ------------------------------------------------------------------
    # NeXus save/load — Phase 2 replacement for save_to_h5/load_from_h5
    # ------------------------------------------------------------------

    def save_to_nexus(self, parent_grp, global_mask=None):
        """Save integration results to a NeXus-formatted HDF5 group.

        Writes into ``parent_grp/<idx>`` (1D) and ``parent_grp/<idx>_2d``
        (2D, optional).  Stores a downsampled thumbnail of the raw image
        in ``parent_grp/<idx>_thumb`` for fast GUI preview.  The thumbnail
        has both arch-level and global masks baked in as NaN pixels.

        Does NOT store full-resolution raw images (map_raw, bg_raw, mask).
        Uses gzip compression only — never lzf (crashes on ARM64 macOS).

        args:
            parent_grp: h5py.Group — typically ``entry/frames/``.
            global_mask: array-like or None — flat indices of detector-level
                masked pixels (from EwaldSphere.global_mask).
        """
        key = f"{self.idx:04d}"

        # ── 1D result ──
        if self.int_1d is not None:
            if key in parent_grp:
                del parent_grp[key]
            grp_1d = parent_grp.create_group(key)
            self.int_1d.to_nexus(grp_1d)
            grp_1d.attrs["source_file"] = self.source_file or ""
            grp_1d.attrs["frame_index"] = int(self.idx)

        # ── Thumbnail (downsampled raw image with mask baked in) ──
        thumb = self.thumbnail
        if thumb is None and self.map_raw is not None:
            try:
                corrected = np.asarray(self.map_raw, dtype=np.float32) - np.asarray(self.bg_raw, dtype=np.float32)
                thumb = _make_thumbnail(corrected, mask_idx=self.mask,
                                        global_mask=global_mask)
            except Exception:
                thumb = None
        if thumb is not None:
            key_thumb = f"{self.idx:04d}_thumb"
            if key_thumb in parent_grp:
                del parent_grp[key_thumb]
            parent_grp.create_dataset(
                key_thumb, data=thumb,
                compression="gzip", compression_opts=1,
            )

        # ── 2D result ──
        if self.int_2d is not None:
            key_2d = f"{self.idx:04d}_2d"
            if key_2d in parent_grp:
                del parent_grp[key_2d]
            grp_2d = parent_grp.create_group(key_2d)
            self.int_2d.to_nexus(grp_2d)
            grp_2d.attrs["source_file"] = self.source_file or ""
            grp_2d.attrs["frame_index"] = int(self.idx)

        # ── GI 1D results ──
        if self.gi_1d:
            for gi_key, res in self.gi_1d.items():
                gi_name = f"{self.idx:04d}_gi1d_{gi_key}"
                if gi_name in parent_grp:
                    del parent_grp[gi_name]
                gi_grp = parent_grp.create_group(gi_name)
                res.to_nexus(gi_grp)
                gi_grp.attrs["source_file"] = self.source_file or ""
                gi_grp.attrs["frame_index"] = int(self.idx)

        # ── GI 2D results ──
        if self.gi_2d:
            for gi_key, res in self.gi_2d.items():
                gi_name = f"{self.idx:04d}_gi2d_{gi_key}"
                if gi_name in parent_grp:
                    del parent_grp[gi_name]
                gi_grp = parent_grp.create_group(gi_name)
                res.to_nexus(gi_grp)
                gi_grp.attrs["source_file"] = self.source_file or ""
                gi_grp.attrs["frame_index"] = int(self.idx)

    def load_from_nexus(self, parent_grp, load_2d=True):
        """Load integration results from a NeXus-formatted HDF5 group.

        Reads from ``parent_grp/<idx>`` (1D), ``parent_grp/<idx>_2d``
        (2D, optional), and ``parent_grp/<idx>_thumb`` (thumbnail).

        args:
            parent_grp: h5py.Group — typically ``entry/frames/``.
            load_2d: bool — if False, skip 2D data and thumbnail.
        """
        key = f"{self.idx:04d}"
        key_2d = f"{self.idx:04d}_2d"
        key_thumb = f"{self.idx:04d}_thumb"

        if key in parent_grp:
            grp_1d = parent_grp[key]
            self.int_1d = IntegrationResult1D.from_hdf5(grp_1d)
            self.source_file = str(grp_1d.attrs.get("source_file", ""))

        # Always load thumbnail (small, useful for preview even in 1D mode)
        if key_thumb in parent_grp:
            self.thumbnail = np.asarray(parent_grp[key_thumb])

        if load_2d and key_2d in parent_grp:
            grp_2d = parent_grp[key_2d]
            self.int_2d = IntegrationResult2D.from_hdf5(grp_2d)

        # GI 1D results
        gi1d_prefix = f"{self.idx:04d}_gi1d_"
        for name in parent_grp:
            if name.startswith(gi1d_prefix):
                gi_key = name[len(gi1d_prefix):]
                self.gi_1d[gi_key] = IntegrationResult1D.from_hdf5(parent_grp[name])

        # GI 2D results
        if load_2d:
            gi2d_prefix = f"{self.idx:04d}_gi2d_"
            for name in parent_grp:
                if name.startswith(gi2d_prefix):
                    gi_key = name[len(gi2d_prefix):]
                    self.gi_2d[gi_key] = IntegrationResult2D.from_hdf5(parent_grp[name])

    def copy(self, include_2d=True):
        """Returns a copy of self.
        """
        arch_copy = EwaldArch(
            copy.deepcopy(self.idx), None,
            copy.deepcopy(self.poni), None,
            copy.deepcopy(self.scan_info), copy.deepcopy(self.ai_args),
            self.file_lock,
            static=copy.deepcopy(self.static), gi=copy.deepcopy(self.gi),
            th_mtr=copy.deepcopy(self.th_mtr),
            sample_orientation=self.sample_orientation,
            series_average=copy.deepcopy(self.series_average)
        )
        arch_copy.integrator = copy.deepcopy(self.integrator)
        arch_copy.arch_lock = Condition()
        arch_copy.int_1d = copy.deepcopy(self.int_1d)
        arch_copy.gi_1d = copy.deepcopy(self.gi_1d)
        arch_copy.source_file = self.source_file
        # Always copy thumbnail — it's small and needed for image preview
        arch_copy.thumbnail = copy.deepcopy(self.thumbnail)
        if include_2d:
            arch_copy.map_raw = copy.deepcopy(self.map_raw)
            arch_copy.bg_raw = copy.deepcopy(self.bg_raw)
            arch_copy.mask = copy.deepcopy(self.mask)
            arch_copy.map_norm = copy.deepcopy(self.map_norm)
            arch_copy.int_2d = copy.deepcopy(self.int_2d)
            arch_copy.gi_2d = copy.deepcopy(self.gi_2d)

        return arch_copy
