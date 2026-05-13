# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:21:58 2019

@author: walroth
"""
import copy
import os
from threading import Condition

from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from pyFAI import units
import numpy as np

from scipy.ndimage import zoom as ndimage_zoom

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
        # Raw-source pointer (R2 schema):
        #   source_file       relpath to the raw source file (image or .nxs master)
        #   source_frame_idx  index of *this* frame within source_file
        # Both written to /entry/frames/frame_NNNN/source/ by the v2
        # writer and round-tripped by _load_arch_v2.  source_frame_idx
        # defaults to None — the writer falls back to ``idx`` for
        # single-frame source files (typical SPEC layout) when it's None.
        self.source_file: str = ""
        self.source_frame_idx: int | None = None
        self.thumbnail: np.ndarray | None = None  # downsampled (map_raw - bg_raw)
        # R3 guardrail: when an arch is reconstructed from a v2 .nxs
        # without a raw image (the common case — v2 doesn't store
        # map_raw to save disk), reintegration is impossible until a
        # lazy raw loader fetches the frame back from ``source_file``.
        # ``_load_arch_v2`` sets ``is_reload_only=False`` when the
        # source path resolves (lazy load will succeed), else ``True``
        # (lazy load can't recover the raw — re-integrate would
        # silently no-op).  The GUI's "Reintegrate" buttons check
        # ``sphere.has_reload_only_frames()`` and pop a message
        # rather than no-op'ing inside ``EwaldArch.integrate_*``.
        self.is_reload_only: bool = False
        # L1 lazy raw load: the directory ``source_file`` resolves
        # against (typically ``os.path.dirname(sphere.data_file)``,
        # set by :func:`_load_arch_v2`).  Empty for fresh-from-wrangler
        # arches that already carry ``map_raw`` and never need lazy
        # load.
        self._source_root: str = ""

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

    def _resolved_source_path(self) -> str:
        """Return the absolute path to ``source_file`` for lazy raw load.

        Joins ``source_file`` (relpath as stored by the wrangler)
        against ``_source_root`` (the sphere's data_file directory,
        set by :func:`_load_arch_v2`).  Returns an empty string when
        either is missing, leaves it to the caller to detect.
        """
        if not self.source_file:
            return ""
        if os.path.isabs(self.source_file):
            return self.source_file
        if not self._source_root:
            # Pre-R2 reloads might lack the root — assume cwd.  Real
            # callers always go through _load_arch_v2 which sets it.
            return self.source_file
        return os.path.normpath(
            os.path.join(self._source_root, self.source_file)
        )

    def _lazy_load_resolvable(self) -> bool:
        """Return True iff a lazy raw load would find the source file.

        Used by :func:`_load_arch_v2` to decide ``is_reload_only`` at
        load time — without it the R3 guardrail would have to either
        always pop (treating every reload as unrecoverable) or never
        pop (silently no-op'ing when the source file is missing).
        """
        full = self._resolved_source_path()
        return bool(full) and os.path.exists(full)

    def _lazy_load_raw(self) -> bool:
        """Best-effort load of ``map_raw`` from the source file on disk.

        Dispatches by source-file extension:

        * ``.h5`` / ``.nxs`` / ``.hdf5`` →
          :class:`ssrl_xrd_tools.io.nexus.NexusImageStack` indexed by
          ``source_frame_idx`` (the wrangler stamped the global stack
          offset, so this works for both single-dataset and Eiger
          multi-link masters).
        * any other extension → ssrl ``read_image`` (TIF/EDF/CBF/…).

        Idempotent: returns immediately if ``map_raw`` is already
        populated.  Returns ``True`` on success, ``False`` on any
        failure (no source ref, file missing, unrecognised format,
        IO error).  Never raises — callers can fall through to the
        existing "no raw, can't integrate" guard.

        Note: this method does **not** acquire ``arch_lock``; callers
        that are already inside the lock (the integrate methods) must
        avoid re-entering.
        """
        if self.map_raw is not None:
            return True
        full = self._resolved_source_path()
        if not full or not os.path.exists(full):
            return False
        ext = os.path.splitext(full)[1].lower()
        try:
            if ext in (".h5", ".nxs", ".hdf5"):
                from ssrl_xrd_tools.io.nexus import open_nexus_image_stack
                # source_frame_idx is the global flattened index per
                # the R2 schema for NeXus sources.
                fidx = self.source_frame_idx
                if fidx is None:
                    fidx = self.idx if self.idx is not None else 0
                with open_nexus_image_stack(full) as stack:
                    self.map_raw = np.asarray(
                        stack[int(fidx)], dtype=np.float32,
                    )
            else:
                from ssrl_xrd_tools.io.image import read_image
                self.map_raw = np.asarray(read_image(full), dtype=np.float32)
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(
                "_lazy_load_raw failed for %s frame %s: %s",
                full, self.source_frame_idx, e,
            )
            return False
        return self.map_raw is not None

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
                # L1: try a lazy raw load from source_file before
                # giving up.  Reloaded-from-v2 arches don't carry
                # map_raw on disk (intentional schema choice) but
                # the wrangler stamps source_file + source_frame_idx,
                # which _lazy_load_raw can use to fetch the original
                # frame from a TIF or a NeXus master.
                self._lazy_load_raw()
            if self.map_raw is None:
                return  # no raw data and no source — cannot re-integrate

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
                # pyFAI 2025.x's fiber integrators do NOT have CSR fast
                # paths for the qip/qoop/qtot/exit spaces — passing
                # method='csr' triggers a "No fast path for space" warning
                # and falls back to a much slower (and visually incorrect)
                # code path.  Keep 'no' here.  The standard transmission
                # integration in arch.integrate_1d (non-GI branch) still
                # uses csr via bai_1d_args['method'].
                gi_method = 'no'

                # Only compute the selected GI 1D mode
                if gi_mode_1d == 'q_ip':
                    # IP profile: integrate over OOP, return Q_ip
                    result = integrate_gi_1d(
                        image_data, fi, npt=numpoints, npt_oop=npt_oop,
                        unit='qip_A^-1',
                        method=gi_method, mask=mask,
                        radial_range=radial_range,
                        azimuth_range=kwargs.get('azimuth_range'),
                        vertical_integration=False,
                        **gi_kwargs,
                    )
                    self.gi_1d['qip'] = result
                elif gi_mode_1d == 'q_oop':
                    # OOP profile: integrate over IP, return Q_oop
                    result = integrate_gi_1d(
                        image_data, fi, npt=numpoints, npt_oop=npt_oop,
                        unit='qoop_A^-1',
                        method=gi_method, mask=mask,
                        radial_range=radial_range,
                        azimuth_range=kwargs.get('azimuth_range'),
                        vertical_integration=True,
                        **gi_kwargs,
                    )
                    self.gi_1d['qoop'] = result
                elif gi_mode_1d == 'exit_angle':
                    result = integrate_gi_exitangles_1d(
                        image_data, fi, npt=numpoints,
                        method=gi_method, mask=mask,
                        radial_range=radial_range,
                        azimuth_range=kwargs.get('azimuth_range'),
                        **gi_kwargs,
                    )
                    self.gi_1d['exit'] = result
                else:  # 'q_total' (default — polar integration)
                    result = integrate_gi_polar_1d(
                        image_data, fi, npt=numpoints,
                        method=gi_method, mask=mask,
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
                # L1: same lazy-load attempt as integrate_1d.
                self._lazy_load_raw()
            if self.map_raw is None:
                return  # no raw data and no source — cannot re-integrate

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
                # See integrate_1d for why GI sticks with method='no':
                # pyFAI 2025.x has no CSR fast-path for qip/qoop spaces.
                gi_method = 'no'

                # Only compute the selected GI 2D mode
                if gi_mode_2d == 'q_chi':
                    result = integrate_gi_polar(
                        image_data, fi, npt_rad=npt_rad, npt_azim=npt_azim,
                        method=gi_method, mask=mask,
                        radial_range=radial_range,
                        azimuth_range=azimuth_range,
                        **gi_kwargs,
                    )
                    self.gi_2d['polar'] = result
                elif gi_mode_2d == 'exit_angles':
                    result = integrate_gi_exitangles(
                        image_data, fi, npt_rad=npt_rad, npt_azim=npt_azim,
                        method=gi_method, mask=mask,
                        radial_range=radial_range,
                        azimuth_range=azimuth_range,
                        **gi_kwargs,
                    )
                    self.gi_2d['exit2d'] = result
                else:  # 'qip_qoop' (default)
                    r_gi2d = integrate_gi_2d(
                        image_data, fi, npt_rad=npt_rad, npt_azim=npt_azim,
                        method=gi_method, mask=mask,
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

    def make_thumbnail(self, global_mask=None):
        """Compute and cache a downsampled (map_raw - bg_raw) preview.

        Stores the result on ``self.thumbnail``; :meth:`save_to_nexus`
        picks it up and skips its own (serial-path) computation.  Safe to
        call from a worker thread — only numpy + scipy work, no HDF5 I/O.

        Parameters
        ----------
        global_mask : array-like or None
            Flat indices of detector-level masked pixels, typically
            ``EwaldSphere.global_mask`` / wrangler ``self.mask``.  Masked
            pixels become NaN in the thumbnail before downsampling.
        """
        if self.thumbnail is not None:
            return
        if self.map_raw is None:
            return
        try:
            corrected = (np.asarray(self.map_raw, dtype=np.float32)
                         - np.asarray(self.bg_raw, dtype=np.float32))
        except Exception:
            return
        self.thumbnail = _make_thumbnail(
            corrected, mask_idx=self.mask, global_mask=global_mask,
        )

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
