"""Streaming direct-beam distance calibration for uniform, flat detectors.

See ``docs/core/direct_beam.md`` for the finite geometry and coordinate contract.
No integration, plotting, GUI, file writes or calibration-asset construction.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import copy
from pathlib import Path
import warnings

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

from xrd_tools.core.geometry.diffractometer import ImageOrientation
from xrd_tools.integrate.calibration import get_detector
from xrd_tools.io.image import read_image


@dataclass(frozen=True)
class DirectBeamResult:
    """One entry per original row; rejected centres/residuals are NaN.

    ``distance_std_m`` is an unweighted regression standard error conditional on
    the chosen model, known angles and pitch; it excludes systematic error.
    ``slope_deg_per_pixel`` is the signed local slope at angle zero for tangent,
    or the fitted notebook slope for linear. Arrays use the oriented frame.
    """

    distance_m: float
    distance_std_m: float
    angles_deg: np.ndarray
    frame_ids: tuple
    point_ids: tuple
    centres_px: np.ndarray
    centre_std_px: np.ndarray
    used: np.ndarray
    rejection_reasons: tuple[str | None, ...]
    predictions_px: np.ndarray
    residuals_px: np.ndarray
    model: str
    angle_zero_deg: float | None
    pixel_pitch_m: float
    image_shape: tuple[int, int]
    slope_deg_per_pixel: float
    intercept_px: float

    @property
    def distance_mm(self) -> float:
        return self.distance_m * 1000.0

    @property
    def motion_sign(self) -> int:
        return int(np.sign(self.slope_deg_per_pixel))


def _uniform_flat(detector):
    # Check BEFORE deepcopy: pyFAI's generic clone omits instance geometry flags.
    if (not detector.IS_FLAT or not detector.uniform_pixel
            or detector.spline is not None):
        raise ValueError('direct-beam calibration requires uniform, flat, undistorted pixels')


def _effective_detector(detector, shape):
    det = copy.deepcopy(detector)
    if det.shape is None:
        det.shape = det.max_shape
    if det.shape is None:
        raise ValueError('detector shape is required (generic: max_shape)')
    if tuple(det.shape) != shape:
        if det.max_shape is None or any(m % n for m,n in zip(det.max_shape,shape)):
            raise ValueError(f'image shape {shape} is not an integer binning of {det.max_shape}')
        if det.force_pixel:
            if not det.guess_binning(shape):
                raise ValueError(f'unsupported detector binning for image shape {shape}')
        else:
            det.binning = tuple(m // n for m,n in zip(det.max_shape,shape))
        if tuple(det.shape) != shape:
            raise ValueError('effective detector shape differs from image shape')
    _uniform_flat(det)
    pitches = np.array([det.pixel1,det.pixel2],dtype=float)
    if not np.all(np.isfinite(pitches) & (pitches > 0)):
        raise ValueError('detector pixel pitches must be finite positive metres')
    return det, pitches


def _gaussian_line(x, background, gradient, amplitude, centre, sigma):
    return background + gradient*x + amplitude*np.exp(-.5*((x-centre)/sigma)**2)


def _peak(image, invalid, beam, motion, half_width, roi, saturation):
    h,w = image.shape
    x0,y0,x1,y1 = (0,0,w,h) if roi is None else roi
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError('ROI must be nonempty, half-open and within the oriented image')
    if not (x0 <= beam[0] < x1 and y0 <= beam[1] < y1):
        raise ValueError('beam_pixel must be inside the oriented image/ROI')
    if motion == 'vertical':
        image, invalid = image.T, invalid.T
        x0,y0,x1,y1 = y0,x0,y1,x1
        beam = beam[::-1]
    row = int(np.floor(beam[1]+.5))
    lo,hi = max(y0,row-half_width), min(y1,row+half_width+1)
    strip = image[lo:hi,x0:x1]
    bad = invalid[lo:hi,x0:x1] | ~np.isfinite(strip) | (strip < 0)
    # Do not mask out saturation and then fit the remaining wings as a valid beam.
    limit = saturation
    if np.issubdtype(image.dtype,np.integer):
        native_limit = np.iinfo(image.dtype).max
        limit = native_limit if limit is None else min(limit,native_limit)
    if limit is not None and np.any(strip >= limit):
        raise ValueError('saturated pixels in beam strip/ROI')
    valid = ~bad.any(axis=0)
    x = np.arange(x0,x1,dtype=float)
    profile = strip.sum(axis=0,dtype=float)  # new array, never a mutable image view
    if valid.sum() < 7:
        raise ValueError('too few valid profile pixels for Gaussian peak')
    xv,yv = x[valid],profile[valid]
    # A linear baseline from the ends keeps the Gaussian seed independent of slope.
    ends = np.r_[0:min(5,len(xv)//3), max(0,len(xv)-5):len(xv)]
    gradient,background = np.polyfit(xv[ends],yv[ends],1)
    signal = yv - (background+gradient*xv)
    amplitude = float(signal.max())
    if amplitude <= max(1e-10, 1e-8*float(np.max(np.abs(yv)))):
        raise ValueError('no positive beam peak above background')
    with warnings.catch_warnings():
        warnings.simplefilter('error',OptimizeWarning)
        fit,cov = curve_fit(_gaussian_line,xv,yv,
            p0=(background,gradient,amplitude,xv[np.argmax(signal)],2.),
            bounds=([-np.inf,-np.inf,0,x0-.5,.25],
                    [np.inf,np.inf,np.inf,x1-.5,(x1-x0)/2]),maxfev=5000)
    _,_,height,centre,sigma = fit
    noise = np.sqrt(np.mean((yv-_gaussian_line(xv,*fit))**2))
    if (not np.all(np.isfinite(cov)) or height <= 5*noise
            or sigma >= (x1-x0)/4 or not x0-1e-6 <= centre <= x1-1+1e-6):
        raise ValueError('poor or unconstrained Gaussian peak fit')
    core = np.abs(x-centre) <= 2*sigma
    if np.any(~valid & core):
        raise ValueError('masked/gap/invalid pixels intersect fitted beam core')
    error = float(np.sqrt(max(0.,cov[3,3])))
    if error > sigma:
        raise ValueError('beam centre uncertainty exceeds peak width')
    return float(centre),error


def calibrate_direct_beam(
    angles_deg: Sequence[float], images: Iterable, *, detector,
    beam_pixel: tuple[float, float], detector_config: Mapping | None = None,
    motion: str = 'horizontal', strip_half_width: int = 2,
    model: str = 'tangent', angle_zero_deg: float | None = None,
    orientation: ImageOrientation | None = None, roi: tuple[int,int,int,int] | None = None,
    mask: np.ndarray | None = None, saturation: float | None = None,
    reader_options: Mapping | None = None, frame_ids: Sequence | None = None,
    point_ids: Sequence | None = None,
) -> DirectBeamResult:
    """Fit distance from explicit degree angles and ordered images/references.

    An item is a 2-D array, path, ``(path, frame)`` or ``(path, frame, dataset)``;
    ``None`` denotes a missing frame. References use ``read_image`` with exact
    frame selection. ``reader_options`` accepts raw dtype/header/shape and HDF
    dataset/frame options, never image transforms or masking.

    ``detector`` is a registered pyFAI name or configured detector. A generic
    name ``'Detector'`` accepts ``detector_config`` with ``pixel1`` (row metres),
    ``pixel2`` (column metres), ``max_shape``. Integer full-panel binning is
    resolved against each scan's first readable image; crops are not binning.
    All frames must then have that shape. Detector objects/images are not mutated.

    ``orientation`` transforms raw arrays and detector masks once. The supplied
    ``beam_pixel``, ``mask`` and half-open ROI ``(x0,y0,x1,y1)`` ALREADY describe
    the oriented image. ``motion`` is also in that frame. Detector_config's
    pyFAI orientation does not apply another array transform.

    Tangent fits centre = c0 + b*tan(angle-angle_zero), requiring an explicit
    known zero; distance = abs(b)*pitch. Only c0 and b are free. The distance is
    perpendicular sample-to-plane separation for a sample-centred detector arm.
    ``model='linear'`` uses the notebook's inverse regression and local distance
    approximation, without needing a known zero. No tilt/wavelength/PONI is fit.
    The beam pixel selects the strip only, not the fitted intercept.

    Negative/nonfinite pixels and masks exclude whole strip-profile bins; invalid
    beam cores and saturation reject a frame. Set ``saturation`` to the detector's
    acquisition limit (inclusive); integer dtype maximum is also rejected.
    Unknown clipping/count-rate effects cannot be inferred from intensity alone.
    At least three usable rows at distinct angles and significant motion are required.
    """
    angles = np.array(angles_deg,dtype=float,copy=True)
    if angles.ndim != 1 or not np.isfinite(angles).all():
        raise ValueError('angles_deg must be a finite one-dimensional array')
    if len(angles) < 3:
        raise ValueError('at least three angle/frame rows are required')
    if np.unique(angles).size < 3:
        raise ValueError('at least three distinct angles are required')
    if model not in {'tangent','linear'}:
        raise ValueError("model must be 'tangent' or 'linear'")
    if model == 'tangent':
        if angle_zero_deg is None or not np.isfinite(angle_zero_deg):
            raise ValueError('tangent model requires finite explicit angle_zero_deg')
        if np.any(np.abs(angles-angle_zero_deg) >= 89):
            raise ValueError('tangent angles must stay within 89 degrees of angle_zero_deg')
    if motion not in {'horizontal','vertical'}:
        raise ValueError("motion must be 'horizontal' or 'vertical'")
    if not isinstance(strip_half_width,(int,np.integer)) or strip_half_width < 0:
        raise ValueError('strip_half_width must be a nonnegative integer')
    beam = np.asarray(beam_pixel,dtype=float)
    if beam.shape != (2,) or not np.isfinite(beam).all():
        raise ValueError('beam_pixel must be finite (x,y)')
    if roi is not None and (len(roi) != 4 or any(not isinstance(v,(int,np.integer)) for v in roi)):
        raise ValueError('ROI must contain four integer bounds')
    if saturation is not None and (not np.isfinite(saturation) or saturation <= 0):
        raise ValueError('saturation must be finite and positive')
    orient = ImageOrientation() if orientation is None else orientation
    if not isinstance(orient,ImageOrientation):
        raise TypeError('orientation must be an ImageOrientation')
    options = dict(reader_options or {})
    allowed = {'frame','detector_shape','raw_dtype','raw_header_skip','dataset_path'}
    if options.keys()-allowed:
        raise ValueError(f'unsupported reader_options: {sorted(options.keys()-allowed)}')
    if detector_config is None:
        det = get_detector(detector)
    else:
        if not isinstance(detector,str):
            raise ValueError('detector_config requires a registered detector name')
        from pyFAI.detectors import detector_factory
        det = detector_factory(detector,config=dict(detector_config))
    _uniform_flat(det)
    ids = None if frame_ids is None else tuple(frame_ids)
    points = tuple(range(len(angles))) if point_ids is None else tuple(point_ids)
    if (ids is not None and len(ids) != len(angles)) or len(points) != len(angles):
        raise ValueError('identifier and angle lengths differ')
    centres = np.full(len(angles),np.nan)
    errors = centres.copy()
    reasons, references = [],[]
    raw_shape = None
    pitch = effective_shape = detector_mask = None
    iterator = iter(images)
    for index in range(len(angles)):
        try:
            item = next(iterator)
        except StopIteration as exc:
            raise ValueError('image and angle lengths differ') from exc
        references.append(index if isinstance(item,np.ndarray) or item is None else item)
        try:
            if item is None:
                raise ValueError('missing frame')
            if isinstance(item,np.ndarray):
                image = item
            else:
                opts = options.copy()
                path = item
                if isinstance(item,tuple):
                    if len(item) not in (2,3):
                        raise ValueError('frame reference must be (path,frame[,dataset])')
                    path,opts['frame'] = item[:2]
                    if len(item) == 3:
                        opts['dataset_path'] = item[2]
                image = read_image(path,preserve_dtype=True,exact_frame=True,**opts)
            if image.ndim != 2 or not all(image.shape):
                raise ValueError('image shape must be nonempty 2-D')
            if raw_shape is None:
                effective,pitches = _effective_detector(det,image.shape)
                raw_shape = image.shape
                detector_mask = effective.mask
                if detector_mask is not None:
                    if detector_mask.shape != raw_shape:
                        raise ValueError('detector mask shape differs from image shape')
                    detector_mask = orient.apply(detector_mask).astype(bool)
                if orient.swaps_axes:
                    pitches = pitches[::-1]
                pitch = float(pitches[1 if motion == 'horizontal' else 0])
            if image.shape != raw_shape:
                raise ValueError(f'image shape {image.shape} differs from {raw_shape}')
            oriented = orient.apply(image)
            effective_shape = oriented.shape
            invalid = np.zeros(effective_shape,bool) if detector_mask is None else detector_mask.copy()
            dummy, tolerance = effective.get_dummies(image)
            if dummy is not None:
                invalid |= np.abs(oriented.astype(np.float32)-dummy) <= tolerance
            if mask is not None:
                if np.shape(mask) != effective_shape:
                    raise ValueError('mask shape must match the oriented image')
                invalid |= np.asarray(mask,dtype=bool)
            centres[index],errors[index] = _peak(oriented,invalid,beam,motion,
                strip_half_width,roi,saturation)
        except (OSError,ValueError,IndexError,KeyError,RuntimeError,OptimizeWarning) as exc:
            reasons.append(f'{type(exc).__name__}: {exc}')
        else:
            reasons.append(None)
        # No scan stack or array-valued frame identifiers are retained.
        item = image = oriented = None
    sentinel = object()
    if next(iterator,sentinel) is not sentinel:
        raise ValueError('image and angle lengths differ')
    used = np.isfinite(centres)
    if used.sum() < 3 or np.unique(angles[used]).size < 3:
        detail = '; '.join(f'row {i}: {r}' for i,r in enumerate(reasons) if r)
        raise ValueError(f'need at least three usable rows at distinct angles; {detail}')
    # Fit two parameters only, centred for conditioning. Equal weights avoid
    # treating exact synthetic/profile fits as infinitely precise observations.
    if model == 'tangent':
        coordinate = np.tan(np.deg2rad(angles-angle_zero_deg))
        x,y = coordinate[used],centres[used]
    else:
        x,y = centres[used],angles[used]
    dx = x-x.mean()
    ss = float(dx@dx)
    if ss <= np.finfo(float).eps*max(1.,float(x@x)):
        raise ValueError('degenerate calibration coordinate range')
    slope = float(dx@(y-y.mean())/ss)
    intercept = float(y.mean()-slope*x.mean())
    regression_residual = y-(intercept+slope*x)
    slope_std = float(np.sqrt((regression_residual@regression_residual)/(len(x)-2)/ss))
    if abs(slope) <= max(1e-12,3*slope_std) or np.ptp(centres[used]) < 1e-6:
        raise ValueError('degenerate or insignificant beam motion')
    if model == 'tangent':
        distance = abs(slope)*pitch
        uncertainty = slope_std*pitch
        predictions = intercept+slope*coordinate
        slope_deg = float(np.rad2deg(1/slope))
        intercept_px = intercept
    else:
        radians = abs(np.deg2rad(slope))
        if radians >= np.pi/2:
            raise ValueError('linear slope is outside the local small-angle regime')
        distance = pitch/np.tan(radians)
        uncertainty = pitch/np.sin(radians)**2*np.deg2rad(slope_std)
        predictions = (angles-intercept)/slope
        slope_deg = slope
        intercept_px = -intercept/slope
    return DirectBeamResult(float(distance),float(uncertainty),angles,
        tuple(references) if ids is None else ids,points,centres,errors,used,
        tuple(reasons),predictions,centres-predictions,model,angle_zero_deg,pitch,
        effective_shape,slope_deg,float(intercept_px))


def calibrate_direct_beam_scan(
    scan_file: str | Path, scan: str | int, *, images: Iterable | None = None,
    image_dir: str | Path | None = None, filename_template: str | None = None,
    point_index_origin: int = 0, motor: str = 'del', **calibration_options,
) -> DirectBeamResult:
    """Thin SPEC adapter; ``scan=23`` means key ``'23.1'``, never list index.

    Supply ordered ``images`` OR ``image_dir`` and an explicit filename template
    with ``{point}`` (and optionally ``{scan}``, ``{repetition}``). Row numbers
    start at ``point_index_origin``; no missing-path discovery or renumbering.
    All numerical/reader options pass to :func:`calibrate_direct_beam`.
    """
    from xrd_tools.io.spec import read_spec_scan_table
    key = str(scan)
    if '.' not in key:
        key += '.1'
    columns,_,count = read_spec_scan_table(scan_file,key)
    if motor not in columns:
        raise ValueError(f'{motor!r} is not a scanned column in SPEC {key}')
    if not isinstance(point_index_origin,int):
        raise ValueError('point_index_origin must be an integer')
    points = tuple((key,i+point_index_origin) for i in range(count))
    if images is None:
        if image_dir is None or filename_template is None or '{point' not in filename_template:
            raise ValueError('supply images or image_dir and filename_template with {point}')
        number,repetition = key.split('.')
        images = (Path(image_dir)/filename_template.format(point=point,scan=int(number),
                   repetition=int(repetition)) for _,point in points)
    elif image_dir is not None or filename_template is not None:
        raise ValueError('supply either images or image_dir/filename_template')
    return calibrate_direct_beam(columns[motor],images,point_ids=points,**calibration_options)


__all__ = ['DirectBeamResult','calibrate_direct_beam','calibrate_direct_beam_scan']
