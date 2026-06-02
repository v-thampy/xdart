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
        Flat indices of pixels to mask (frame-level mask).
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


class LiveFrame():
    """Stateful xdart detector frame collected in
    X-ray diffraction experiments.

    Attributes:
        ai_args: dict, arguments passed to AzimuthalIntegrator
        frame_lock: Condition, threading lock used to ensure only one
            process can access data at a time
        file_lock: Condition, lock to ensure only one writer to
            data file
        idx: int, integer name of frame
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
        copy: create copy of frame
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
                 scan_info=None, ai_args=None, file_lock=None,
                 static=False, bg_raw=0,
                 gi=False, incidence_motor=None, tilt_angle=0,
                 sample_orientation=4,
                 series_average=False, integrator=None,
                 *, th_mtr=None,  # J1: legacy alias, mapped to incidence_motor
                 ):
        # pylint: disable=too-many-arguments
        """idx: int, name of the frame.
        map_raw: numpy array, raw image data
        bg_raw: numpy array, raw image data for BG
        poni: PONI object, calibration data
        mask: None or numpy array, indices of pixels to mask
        scan_info: dict, metadata about scan (None → fresh dict per instance)
        ai_args: dict, args to be fed to azimuthalIntegrator constructor
        file_lock: Condition, lock for file access (None → fresh Condition).
        incidence_motor: str, name of the GI incidence-angle motor
        (or its scan_info key).  ``th_mtr`` is accepted as a legacy
        alias kwarg and mapped here; ``self.th_mtr`` remains a
        property alias for backward read access.
        """
        super().__init__()
        # F4: None-sentinel pattern.  Mutable defaults in the signature
        # (scan_info={}, ai_args={}, file_lock=Condition()) are
        # constructed once at module-import time and shared by every
        # caller who omits the kwarg — silently producing cross-frame
        # state.  Materialise per-instance defaults here, matching
        # what LiveScan already does.
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
        self.scan_info = scan_info if scan_info is not None else {}
        self.ai_args = ai_args if ai_args is not None else {}
        self.file_lock = file_lock if file_lock is not None else Condition()

        self.static = static
        self.gi = gi
        # J1: canonical name is ``incidence_motor``.  Pick up the
        # value from either kwarg with ``incidence_motor=`` winning
        # if both happen to be passed (shouldn't be — but be
        # deterministic).  Default 'th' preserves the historical
        # frame behavior.
        if incidence_motor is None:
            incidence_motor = th_mtr if th_mtr is not None else 'th'
        self.incidence_motor = incidence_motor
        self.tilt_angle = tilt_angle
        self.sample_orientation = sample_orientation
        self.series_average = series_average

        self.integrator = integrator if integrator is not None else self.setup_integrator()
        # Disable pyFAI's legacy mask heuristic that auto-inverts masks
        # with > 50% masked pixels (e.g. threshold masks).
        self.integrator.USE_LEGACY_MASK_NORMALIZATION = False

        self.frame_lock = Condition()
        self.map_norm = 1

        self.int_1d: IntegrationResult1D | None = None
        self.int_2d: IntegrationResult2D | None = None
        self.gi_1d: dict[str, IntegrationResult1D] = {}
        self.gi_2d: dict[str, IntegrationResult2D] = {}
        # Raw-source pointer (R2 schema):
        #   source_file       relpath to the raw source file (image or .nxs master)
        #   source_frame_idx  index of *this* frame within source_file
        # Both written to /entry/frames/frame_NNNN/source/ by the v2
        # writer and round-tripped by _load_frame_v2.  source_frame_idx
        # defaults to None — the writer falls back to ``idx`` for
        # single-frame source files (typical SPEC layout) when it's None.
        self.source_file: str = ""
        self.source_frame_idx: int | None = None
        self.thumbnail: np.ndarray | None = None  # downsampled (map_raw - bg_raw)
        # R3 guardrail: when an frame is reconstructed from a v2 .nxs
        # without a raw image (the common case — v2 doesn't store
        # map_raw to save disk), reintegration is impossible until a
        # lazy raw loader fetches the frame back from ``source_file``.
        # ``_load_frame_v2`` sets ``is_reload_only=False`` when the
        # source path resolves (lazy load will succeed), else ``True``
        # (lazy load can't recover the raw — re-integrate would
        # silently no-op).  The GUI's "Reintegrate" buttons check
        # ``scan.has_reload_only_frames()`` and pop a message
        # rather than no-op'ing inside ``LiveFrame.integrate_*``.
        self.is_reload_only: bool = False
        # L1 lazy raw load: the directory ``source_file`` resolves
        # against (typically ``os.path.dirname(scan.data_file)``,
        # set by :func:`_load_frame_v2`).  Empty for fresh-from-wrangler
        # frames that already carry ``map_raw`` and never need lazy
        # load.
        self._source_root: str = ""

    def __getstate__(self):
        """Exclude threading.Condition objects so LiveFrame can be pickled
        for use with concurrent.futures.ProcessPoolExecutor."""
        state = self.__dict__.copy()
        state.pop('frame_lock', None)
        state.pop('file_lock', None)
        return state

    def __setstate__(self, state):
        """Restore threading.Condition objects after unpickling."""
        self.__dict__.update(state)
        self.frame_lock = Condition()
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
        against ``_source_root`` (the scan's data_file directory,
        set by :func:`_load_frame_v2`).  Returns an empty string when
        either is missing, leaves it to the caller to detect.
        """
        if not self.source_file:
            return ""
        if os.path.isabs(self.source_file):
            return self.source_file
        if not self._source_root:
            # Pre-R2 reloads might lack the root — assume cwd.  Real
            # callers always go through _load_frame_v2 which sets it.
            return self.source_file
        return os.path.normpath(
            os.path.join(self._source_root, self.source_file)
        )

    def _lazy_load_resolvable(self) -> bool:
        """Return True iff a lazy raw load would find the source file.

        Used by :func:`_load_frame_v2` to decide ``is_reload_only`` at
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

        Note: this method does **not** acquire ``frame_lock``; callers
        that are already inside the lock (the integrate methods) must
        avoid re-entering.
        """
        if self.map_raw is not None:
            return True
        full = self._resolved_source_path()
        if not full or not os.path.exists(full):
            return False
        ext = os.path.splitext(full)[1].lower()
        # source_frame_idx is the global frame offset within the
        # source file per the R2 schema.  Resolve once so both the
        # HDF5 and non-HDF5 branches can pass the same value.  For
        # single-frame TIFF/EDF/CBF the value is 0 (the writer
        # stamps it that way), so ``frame=0`` is a no-op; for
        # multi-frame TIFF/CBF stacks the wrangler records the
        # correct offset and we must forward it.
        fidx = self.source_frame_idx
        if fidx is None:
            fidx = self.idx if self.idx is not None else 0
        fidx = int(fidx)
        try:
            if ext in (".h5", ".nxs", ".hdf5"):
                from ssrl_xrd_tools.io.nexus import open_nexus_image_stack
                with open_nexus_image_stack(full) as stack:
                    self.map_raw = np.asarray(
                        stack[fidx], dtype=np.float32,
                    )
            else:
                # O2: forward source_frame_idx for multi-frame
                # non-HDF5 sources too (e.g. multi-frame TIFF /
                # CBF stacks).  Single-frame files use frame=0
                # and ignore the kwarg, so this is safe.
                from ssrl_xrd_tools.io.image import read_image
                self.map_raw = np.asarray(
                    read_image(full, frame=fidx), dtype=np.float32,
                )
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug(
                "_lazy_load_raw failed for %s frame %s: %s",
                full, self.source_frame_idx, e,
            )
            return False
        return self.map_raw is not None

    @property
    def th_mtr(self):
        """Legacy alias for :attr:`incidence_motor`.

        J1: deprecated.  Kept for read access from old GUI signal
        handlers that pass the motor name positionally.  New code
        should use ``self.incidence_motor`` directly.
        """
        return self.incidence_motor

    @th_mtr.setter
    def th_mtr(self, value):
        self.incidence_motor = value

    def _get_incident_angle(self):
        """Return incident angle in degrees from incidence_motor or scan_info."""
        try:
            return float(self.incidence_motor)
        except (ValueError, TypeError):
            # Case-insensitive lookup for motor position
            motor = str(self.incidence_motor).lower()
            for key, val in self.scan_info.items():
                if key.lower() == motor:
                    return float(val)
            return 0.0

    def reset(self):
        """Clears all data, resets to a default LiveFrame.
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
            
    def _resolve_monitor_norm(self, monitor) -> float:
        """Resolve the monitor normalization scalar for this frame.

        Centralises the lookup that ``integrate_1d`` and ``integrate_2d``
        do independently — pre-fix ``integrate_2d`` set ``map_norm = 1``
        unconditionally when ``monitor`` was provided, silently
        skipping normalization, and never reset it when ``monitor``
        was None (so a prior ``integrate_1d`` value leaked through).

        Behaviour:

        * ``monitor is None`` → return 1.0 (no normalization).  Always
          reset state — don't inherit from a previous call.
        * ``monitor.upper()`` in ``scan_info`` → use that value.
        * ``monitor.lower()`` in ``scan_info`` → use that value (back-
          compat with lowercase counter names some SPEC setups use).
        * monitor not found → return 1.0 and log debug.

        Returns the resolved scalar; callers are expected to assign
        it to ``self.map_norm`` so the integrator divides by it.
        """
        if monitor is None:
            return 1.0
        if monitor.upper() in self.scan_info:
            return float(self.scan_info[monitor.upper()])
        if monitor.lower() in self.scan_info:
            return float(self.scan_info[monitor.lower()])
        import logging as _logging
        _logging.getLogger(__name__).debug(
            "monitor=%r not in scan_info keys=%s; using 1.0",
            monitor, list(self.scan_info.keys()),
        )
        return 1.0

    def get_mask(self, global_mask=None):
        shape = self.map_raw.shape
        size = self.map_raw.size
        mask = np.zeros(size, dtype=bool)
        # Apply the frame's own bad-pixel mask (always derived from this
        # image, so in-bounds) first.
        if self.mask is not None and len(self.mask):
            try:
                mask[self.mask] = True
            except IndexError:
                import logging
                logging.getLogger(__name__).warning(
                    "Ignoring frame mask: indices out of bounds for image "
                    "size %d.", size,
                )
        # The external detector/global mask may have been built for a
        # different detector shape (wrong calibration, resized frame, …).
        # Ignore it with a warning rather than raising — reducing unmasked
        # beats aborting the scan.  Mirrors reduction._flat_mask_as_bool.
        if global_mask is not None and len(global_mask):
            gm = np.asarray(global_mask)
            incompatible = (
                (gm.dtype == bool and gm.size != size)
                or (gm.dtype != bool and gm.size
                    and (gm.min() < 0 or gm.max() >= size))
            )
            if incompatible:
                import logging
                logging.getLogger(__name__).warning(
                    "Ignoring detector/global mask: incompatible with image "
                    "shape %s (image has %d pixels).", shape, size,
                )
            else:
                mask[gm.ravel() if gm.dtype == bool else gm] = True
        return mask.reshape(shape)

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
        with self.frame_lock:
            if self.map_raw is None:
                # L1: try a lazy raw load from source_file before
                # giving up.  Reloaded-from-v2 frames don't carry
                # map_raw on disk (intentional schema choice) but
                # the wrangler stamps source_file + source_frame_idx,
                # which _lazy_load_raw can use to fetch the original
                # frame from a TIF or a NeXus master.
                self._lazy_load_raw()
            if self.map_raw is None:
                return  # no raw data and no source — cannot re-integrate

            # Q1 (post-0.37.1 review): use the shared helper instead of
            # the inline scan_info lookup so 1D and 2D integration paths
            # agree on what "monitor=…" means.  ``map_norm`` is always
            # reset (no inheritance from a prior call's value).
            self.map_norm = self._resolve_monitor_norm(monitor)

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
                # integration in frame.integrate_1d (non-GI branch) still
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
        with self.frame_lock:
            if self.map_raw is None:
                # L1: same lazy-load attempt as integrate_1d.
                self._lazy_load_raw()
            if self.map_raw is None:
                return  # no raw data and no source — cannot re-integrate

            # Q1: 2D normalization was effectively broken — pre-fix
            # this set ``map_norm = 1`` (no normalization) whenever
            # ``monitor`` was provided, and didn't reset state when
            # ``monitor`` was None.  Use the shared helper so 1D
            # and 2D agree on the lookup.
            self.map_norm = self._resolve_monitor_norm(monitor)

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

        with self.frame_lock:
            self.ai_args = args
            self.integrator = self.setup_integrator()

    def set_map_raw(self, new_data):
        with self.frame_lock:
            self.map_raw = new_data
            if self.mask is None:
                self.mask = np.arange(new_data.size)[new_data.flatten() < 0]

    def set_poni(self, new_data):
        with self.frame_lock:
            self.poni = new_data

    def set_mask(self, new_data):
        with self.frame_lock:
            self.mask = new_data

    def set_scan_info(self, new_data):
        with self.frame_lock:
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
            ``LiveScan.global_mask`` / wrangler ``self.mask``.  Masked
            pixels become NaN in the thumbnail before downsampling.
        """
        if self.thumbnail is not None:
            return
        if self.map_raw is None:
            # F5: try L1 lazy load before giving up.  Lets thumbnail
            # regeneration work on reloaded v2 scans whose source
            # files are still available.
            self._lazy_load_raw()
        if self.map_raw is None:
            return
        try:
            corrected = (np.asarray(self.map_raw, dtype=np.float32)
                         - np.asarray(self.bg_raw, dtype=np.float32))
        except (TypeError, ValueError) as e:
            import logging
            logging.getLogger(__name__).debug(
                "make_thumbnail: bg subtraction failed: %s", e,
            )
            return
        self.thumbnail = _make_thumbnail(
            corrected, mask_idx=self.mask, global_mask=global_mask,
        )

    def copy(self, include_2d=True):
        """Returns a copy of self.
        """
        frame_copy = LiveFrame(
            copy.deepcopy(self.idx), None,
            copy.deepcopy(self.poni), None,
            copy.deepcopy(self.scan_info), copy.deepcopy(self.ai_args),
            self.file_lock,
            static=copy.deepcopy(self.static), gi=copy.deepcopy(self.gi),
            incidence_motor=copy.deepcopy(self.incidence_motor),
            sample_orientation=self.sample_orientation,
            series_average=copy.deepcopy(self.series_average)
        )
        frame_copy.integrator = copy.deepcopy(self.integrator)
        frame_copy.frame_lock = Condition()
        frame_copy.int_1d = copy.deepcopy(self.int_1d)
        frame_copy.gi_1d = copy.deepcopy(self.gi_1d)
        # P4: lazy-raw-load provenance.  Pre-P4 ``source_file`` was
        # copied but ``source_frame_idx``, ``_source_root``, and
        # ``is_reload_only`` were not — so an ``frame.copy(include_2d=False)``
        # stash in ``data_1d`` couldn't recover ``map_raw`` via
        # :meth:`_lazy_load_raw` later (it'd resolve to the wrong
        # frame or be unable to resolve a relpath at all).  Copy
        # all four so a 1D-only copy retains the ability to lazy-
        # load the raw frame on demand (reintegrate, thumbnail
        # regeneration, etc.).
        frame_copy.source_file = self.source_file
        frame_copy.source_frame_idx = self.source_frame_idx
        frame_copy._source_root = self._source_root
        frame_copy.is_reload_only = self.is_reload_only
        # Always copy thumbnail — it's small and needed for image preview
        frame_copy.thumbnail = copy.deepcopy(self.thumbnail)
        if include_2d:
            frame_copy.map_raw = copy.deepcopy(self.map_raw)
            frame_copy.bg_raw = copy.deepcopy(self.bg_raw)
            frame_copy.mask = copy.deepcopy(self.mask)
            frame_copy.map_norm = copy.deepcopy(self.map_norm)
            frame_copy.int_2d = copy.deepcopy(self.int_2d)
            frame_copy.gi_2d = copy.deepcopy(self.gi_2d)

        return frame_copy

    def copy_for_display(self, include_2d=False):
        """Return a lightweight cache copy for GUI display dictionaries.

        Unlike :meth:`copy`, this does not construct or deep-copy a pyFAI
        integrator.  Display paths treat the cached frame as read-only and
        only need metadata, thumbnails, and reduced results.
        """
        frame_copy = copy.copy(self)
        frame_copy.frame_lock = Condition()
        frame_copy.scan_info = dict(self.scan_info)
        frame_copy.ai_args = dict(self.ai_args)
        frame_copy.gi_1d = dict(self.gi_1d)
        frame_copy.thumbnail = copy.copy(self.thumbnail)
        if include_2d:
            frame_copy.gi_2d = dict(self.gi_2d)
        else:
            frame_copy.map_raw = None
            frame_copy.bg_raw = 0
            frame_copy.mask = None
            frame_copy.int_2d = None
            frame_copy.gi_2d = {}
        return frame_copy


__all__ = ["LiveFrame"]
